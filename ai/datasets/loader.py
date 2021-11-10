import pandas as pd
import os
import json
from pathlib import Path
import kaggle

from sklearn.model_selection import train_test_split

dataset_path = Path(__file__).parent / '../../datasets/'
memes_file = Path(__file__).parent / '../../datasets/memotion_dataset_7k/labels.csv'

#TODO find good dataset
def import_memes(refresh=False):
    if os.path.isfile(memes_file) and not refresh:
        print('### memes.csv already exists ###')
        return

    # kaggle datasets download -d williamscott701/memotion-dataset-7k
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('williamscott701/memotion-dataset-7k', path=dataset_path,
                                      unzip=True)


def get_memes_dataframe():
    return pd.read_csv(memes_file, sep='\t')


def train_validate_test_split(df, label_column, train_percent=.6, validate_percent=.4, train_test_ratio=0.5):
    X = df.drop([label_column], axis=1)
    Y = df[label_column]
    X_train, x, y_train, y = train_test_split(X, Y, test_size=train_percent, train_size=validate_percent)
    X_val, X_test, y_val, y_test = train_test_split(x, y, test_size=train_test_ratio, train_size=1 - train_test_ratio)
    return X_train, y_train, X_val, y_val, X_test, y_test
