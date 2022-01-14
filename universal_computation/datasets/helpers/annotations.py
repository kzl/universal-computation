import pandas as pd
import os
import pathlib
from pathlib import Path

data = 'data/2750'

df = pd.DataFrame(columns=['label', 'int_label', 'img_name'])
labels_dict = {}
counter = 0
for subdir in os.listdir(data):
    labels_dict[subdir] = counter
    filepath = os.path.join(data,subdir)
    if os.path.isdir(filepath):
        for file in os.listdir(filepath):
            dict = {'label': subdir, 'int_label': labels_dict[subdir], 'img_name': file}
            df = df.append(dict, ignore_index = True)
    counter += 1
train = df.sample(frac=0.75,random_state=200) #random state is a seed value
test = df.drop(train.index)

train.to_csv(f'{data}/train.csv')
test.to_csv(f'{data}/test.csv')
