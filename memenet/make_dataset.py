from img2vec_pytorch import Img2Vec
from gensim.models.doc2vec import Doc2Vec

img2vec = Img2Vec(cuda=False)
doc2vec = Doc2Vec.load('memenet/data/doc2vec_wiki_d300_n5_w8_mc50_t12_e10_dbow.model')

#img_emb = img2vec.get_vec(<list of PIL images>)
#txt_emb = doc2vec.infer_vector(<list of words (strings)>)
#save image and text embeddings to files in data/
