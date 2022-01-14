import pandas as pd

def read_annotations(ann_file):
    site_name = pd.read_csv(ann_file, nrows=1, header=None)[0].tolist()[0].split('# Site: ')[1]
    labels_dict = {}
    with open(ann_file, 'r') as f:
        start_reading = False
        for line in f:
            if start_reading:
                if line[0] != '#':
                    break
                else:
                    int_label, str_label = line[1:].split('. ')
                    int_label = int(int_label)
                    str_label = str_label.strip()
                    labels_dict[str_label] = int_label
            if line == '# Categories:\n':
                start_reading = True

    df = pd.read_csv(ann_file, comment='#')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df['label'] = df['label'].astype('category')
    df['int_label'] = [labels_dict[x] for x in df['label']]

    img_name_col = []
    for ts in df.index:
        year = ts[:4]
        month = ts[5:7]
        day = ts[8:10]
        hms = ts.split(' ')[1].replace(':', '')
        img_name_col.append(f'{site_name}_{year}_{month}_{day}_{hms}.jpg')
    df['img_name'] = img_name_col

    return df
