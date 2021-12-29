import pandas as pd
import kaggle
from PIL import Image
from sklearn.model_selection import train_test_split

from ai.datasets.helper import download_img, strip_and_lower, prune_data
from ai.datasets.plotter import show_plots

datasets_path = '../datasets'
kaggle_img_path = datasets_path + '/memes_reference_data.tsv'
kaggle_text_path = datasets_path + '/memes_data.tsv'

memes_file = datasets_path + '/memes_txt/memes_data.csv'
memes_path = datasets_path + '/memes_img/'


def import_memes():
    dataset = 'abhishtagatya/imgflipscraped-memes-caption-dataset'
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path='../datasets',
                                      unzip=True)


def download_images():
    data = pd.read_csv(kaggle_img_path, sep='\t')
    img_data = data[['BaseImageURL', 'MemeLabel']]
    img_data.apply(lambda row: download_img(row['BaseImageURL'], strip_and_lower(row['MemeLabel'])), axis=1)


def prepare_text(with_plots=False):
    data = pd.read_csv(kaggle_text_path, sep='\t')
    data = data.drop(columns=['ImageURL', 'AltText', 'HashId'])
    prune_data(data)
    data.to_csv(memes_file, sep=',', index=False)

    if with_plots:
        show_plots(data)


def get_memes_dataframe():
    return pd.read_csv(memes_file, sep='\t')


def get_PIL_images():
    pil_images_list = []

    import glob
    for filename in glob.glob(memes_path + '/*.jpg'):
        im = Image.open(filename)
        pil_images_list.append(im)

    return pil_images_list


def train_validate_test_split(df, label_column, train_percent=.6, validate_percent=.4, train_test_ratio=0.5):
    X = df.drop([label_column], axis=1)
    Y = df[label_column]
    X_train, x, y_train, y = train_test_split(X, Y, test_size=train_percent, train_size=validate_percent)
    X_val, X_test, y_val, y_test = train_test_split(x, y, test_size=train_test_ratio, train_size=1 - train_test_ratio)
    return X_train, y_train, X_val, y_val, X_test, y_test
