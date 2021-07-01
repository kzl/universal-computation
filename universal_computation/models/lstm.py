"""
MIT License

Copyright (c) 2018 Alex

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Code modified from Github repo: https://github.com/exe1023/LSTM_LN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()


class LNLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.,
                 bidirectional=1,
                 batch_first=False,
                 residual=False,
                 cln=False):
        super(LNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = bidirectional + 1
        self.batch_first = batch_first
        self.residual = residual

        layers = []
        for i in range(num_layers):
            for j in range(self.direction):
                layer = LayerNormLSTM(input_size*self.direction,
                                      hidden_size,
                                      dropout=dropout,
                                      cln=cln)
                layers.append(layer)
            input_size = hidden_size
        self.layers = layers
        self.params = nn.ModuleList(layers)

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def init_hidden(self, batch_size):
        # Uses Xavier init here.
        hiddens = []
        for l in self.layers:
            std = math.sqrt(2.0 / (l.input_size + l.hidden_size))
            h = Variable(Tensor(1, batch_size, l.hidden_size).normal_(0, std))
            c = Variable(Tensor(1, batch_size, l.hidden_size).normal_(0, std))
            if use_cuda:
                hiddens.append((h.cuda(), c.cuda()))
            else:
                hiddens.append((h, c))
        return hiddens

    def layer_forward(self, l, xs, h, image_emb, reverse=False):
        '''
        return:
            xs: (seq_len, batch, hidden)
            h: (1, batch, hidden)
        '''
        if self.batch_first:
            xs = xs.permute(1, 0, 2).contiguous()
        ys = []
        for i in range(xs.size(0)):
            if reverse:
                x = xs.narrow(0, (xs.size(0)-1)-i, 1)
            else:
                x = xs.narrow(0, i, 1)
            y, h = l(x, h, image_emb)
            ys.append(y)
        y = torch.cat(ys, 0)
        if self.batch_first:
            y = y.permute(1, 0, 2)
        return y, h

    def forward(self, x, hiddens=None, image_emb=None):
        if hiddens is None:
            hiddens = self.init_hidden(x.shape[0])
        if self.direction > 1:
            x = torch.cat((x, x), 2)
        if type(hiddens) != list:
            # when the hidden feed is (direction * num_layer, batch, hidden)
            tmp = []
            for idx in range(hiddens[0].size(0)):
                tmp.append((hiddens[0].narrow(0, idx, 1),
                           (hiddens[1].narrow(0, idx, 1))))
            hiddens = tmp

        new_hs = []
        new_cs = []
        for l_idx in range(0, len(self.layers), self.direction):
            l, h = self.layers[l_idx], hiddens[l_idx]
            f_x, f_h = self.layer_forward(l, x, h, image_emb)
            if self.direction > 1:
                l, h  = self.layers[l_idx+1], hiddens[l_idx+1]
                r_x, r_h = self.layer_forward(l, x, h, image_emb, reverse=True)

                x = torch.cat((f_x, r_x), 2)
                h = torch.cat((f_h[0], r_h[0]), 0)
                c = torch.cat((f_h[1], r_h[1]), 0)
            else:
                if self.residual:
                    x = x + f_x
                else:
                    x = f_x
                h, c = f_h
            new_hs.append(h)
            new_cs.append(c)

        h = torch.cat(new_hs, 0)
        c = torch.cat(new_cs, 0)

        return x, (h, c)


class CLN(nn.Module):
    """
    Conditioned Layer Normalization
    """
    def __init__(self, input_size, image_size, epsilon=1e-6):
        super(CLN, self).__init__()
        self.input_size = input_size
        self.image_size = image_size
        self.alpha = Tensor(1, input_size).fill_(1)
        self.beta = Tensor(1, input_size).fill_(0)
        self.epsilon = epsilon

        self.alpha = Parameter(self.alpha)
        self.beta = Parameter(self.beta)

        # MLP used to predict delta of alpha, beta
        self.fc_alpha = nn.Linear(self.image_size, self.input_size)
        self.fc_beta = nn.Linear(self.image_size, self.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def create_cln_input(self, image_emb):
        delta_alpha = self.fc_alpha(image_emb)
        delta_beta = self.fc_beta(image_emb)
        return delta_alpha, delta_beta

    def forward(self, x, image_emb):
        if image_emb is None:
            return x
        # x: (batch, input_size)
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).unsqueeze(1).expand_as(x)) / torch.sqrt(torch.var(x, 1).unsqueeze(1).expand_as(x) + self.epsilon)

        delta_alpha, delta_beta = self.create_cln_input(image_emb)
        alpha = self.alpha.expand_as(x) + delta_alpha
        beta = self.beta.expand_as(x) + delta_beta
        x =  alpha * x + beta
        return x.view(size)


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = Tensor(1, input_size).fill_(1)
        self.beta = Tensor(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).unsqueeze(1).expand_as(x)) / torch.sqrt(torch.var(x, 1).unsqueeze(1).expand_as(x) + self.epsilon)
        if self.learnable:
            x =  self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


class LSTMcell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(LSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = Variable(torch.bernoulli(Tensor(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(torch.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        h_t = torch.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(th.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)


class LayerNormLSTM(LSTMcell):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 bias=True,
                 dropout=0.0,
                 dropout_method='pytorch',
                 ln_preact=True,
                 learnable=True,
                 cln=True):
        super(LayerNormLSTM, self).__init__(input_size=input_size,
                                            hidden_size=hidden_size,
                                            bias=bias,
                                            dropout=dropout,
                                            dropout_method=dropout_method)
        self.cln = cln
        if ln_preact:
            if self.cln:
                self.ln_i2h = CLN(4*hidden_size, 1024)
                self.ln_h2h = CLN(4*hidden_size, 1024)
            else:
                self.ln_h2h = LayerNorm(4*hidden_size, learnable=learnable)
                self.ln_i2h = LayerNorm(4*hidden_size, learnable=learnable)
        self.ln_preact = ln_preact
        if self.cln:
            self.ln_cell = CLN(hidden_size, 1024)
        else:
            self.ln_cell = LayerNorm(hidden_size, learnable=learnable)

    def forward(self, x, hidden, image_emb=None):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln_preact:
            if self.cln:
                i2h = self.ln_i2h(i2h, image_emb)
                h2h = self.ln_h2h(h2h, image_emb)
            else:
                i2h = self.ln_i2h(i2h)
                h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(torch.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        if self.cln:
            c_t = self.ln_cell(c_t, image_emb)
        else:
            c_t = self.ln_cell(c_t)
        h_t = torch.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(torch.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)
