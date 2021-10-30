import urllib.request
import zipfile

cornell_movie_dialogs_corpus = 'www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
handler, _ = urllib.request.urlretrieve(cornell_movie_dialogs_corpus)
zip_file_object = zipfile.ZipFile(handler, 'r')
first_file = zip_file_object.namelist()[0]
file = zip_file_object.open(first_file)
content = file.read()
