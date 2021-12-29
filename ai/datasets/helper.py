import requests
from nltk.tokenize import RegexpTokenizer


def download_img(image_url, img_name):
    img_data = requests.get(image_url).content
    with open('../datasets/memes_img/' + img_name + '.jpg', 'wb') as handler:
        handler.write(img_data)


def strip_and_lower(txt):
    return txt.replace(" ", "").lower()


def prune_data(df):
    tokenizer = RegexpTokenizer(r'\w+')

    df['MemeLabel'] = df.apply(lambda row: strip_and_lower(row['MemeLabel']), axis=1)
    df['CaptionText'] = df.apply(lambda row: tokenizer.tokenize(row['CaptionText'].lower()), axis=1)
    return df
