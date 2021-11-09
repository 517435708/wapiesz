import pandas as pd
import os
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

imgflip_dir = Path(__file__).parent / '../../ImgFlip575K_Dataset/dataset/memes/'
memes_file = Path(__file__).parent / '../../datasets/memes.csv'

#TODO find good dataset
def import_memes(refresh=False):
    if os.path.isfile(memes_file) and not refresh:
        print('### memes.csv already exists ###')
        return

    file_list = [pos_json for pos_json in os.listdir(imgflip_dir) if pos_json.endswith('.json')]
    dfs = []

    for file in file_list:
        with open(os.path.join(imgflip_dir / file), 'r') as json_data:
            data = json.load(json_data)
            df = pd.DataFrame(data)
            A = pd.json_normalize(df['metadata'])

            df = pd.concat([df, A], axis=1)
            df = df.drop(['url', 'post', 'metadata', 'views', 'img-votes', 'author'], axis=1)

            dfs.append(df)  # append the data frame to the list
            break

    temp = pd.concat(dfs, ignore_index=True)  #
    temp.to_csv('../datasets/memes.csv', sep='\t')


def get_memes_dataframe():
    return pd.read_csv(memes_file, sep='\t')


def train_validate_test_split(df, label_column, train_percent=.6, validate_percent=.4, train_test_ratio=0.5):
    X = df.drop([label_column], axis=1)
    Y = df[label_column]
    X_train, x, y_train, y = train_test_split(X, Y, test_size=train_percent, train_size=validate_percent)
    X_val, X_test, y_val, y_test = train_test_split(x, y, test_size=train_test_ratio, train_size=1 - train_test_ratio)
    return X_train, y_train, X_val, y_val, X_test, y_test
