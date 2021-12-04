import requests


def download_img(image_url, img_name):
    img_data = requests.get(image_url).content
    with open('../datasets/memes_img/' + img_name + '.jpg', 'wb') as handler:
        handler.write(img_data)


def strip_and_lower(txt):
    return txt.replace(" ", "").lower()
