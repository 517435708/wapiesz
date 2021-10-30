from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import argparse


def download_and_unzip(https_url, destination='.'):
    http_response = urlopen(https_url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=destination)


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", required=True,
                    help="pass dataset URL, it must be ZIP file!")
extract_to = parser.add_argument("-e", "--extract_to", required=False,
                                 help="pass directory where to extract dataset")

args = vars(parser.parse_args())

url = args['url']
extract_to = args['extract_to']

download_and_unzip(https_url=url, destination=extract_to)
