import pandas as pd
import os
import pathlib
from pathlib import Path

cur_path = pathlib.Path().resolve()
temp = cur_path.parent.absolute()
comp_dir = temp.parent.absolute()
data = str(comp_dir) + '/data/2750'

df = pd.DataFrame(columns=['label', 'int_label', 'img_name'])
labels_dict = {}
counter = 0
for subdir in os.listdir(data):
    labels_dict[subdir] = counter
    counter += 1
    filepath = os.path.join(data,subdir)
    if os.path.isdir(filepath):
        for file in os.listdir(filepath):
            dict = {'label': subdir, 'int_label': labels_dict[subdir], 'img_name': file}
            df = df.append(dict, ignore_index = True)
train=df.sample(frac=0.75,random_state=200) #random state is a seed value
test=df.drop(train.index)

train.to_csv(data+'/train.csv')
test.to_csv(data+'/test.csv')
