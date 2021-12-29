from img2vec_pytorch import Img2Vec
from gensim.models.doc2vec import Doc2Vec
import ai.datasets.loader as loader

img2vec = Img2Vec(cuda=False)
# doc2vec = Doc2Vec.load('memenet/data/doc2vec_wiki_d300_n5_w8_mc50_t12_e10_dbow.model')

list_of_pil_images = loader.get_PIL_images()
list_of_words = loader.get_memes_dataframe()["MemeLabel"]


for img in list_of_pil_images:
	imgvec = img2vec.get_vec(img)
	print(imgvec)
# img_emb = img2vec.get_vec(<list of PIL images>)
# txt_emb = doc2vec.infer_vector(<list of words (strings)>)
# save image and text embeddings to files in data/
