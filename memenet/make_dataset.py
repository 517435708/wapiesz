from img2vec_pytorch import Img2Vec
from gensim.models.doc2vec import Doc2Vec
import ai.datasets.loader as loader
import pandas as pd

prepare_img_vec = False
prepare_txt_vec = True

if prepare_img_vec:
    img2vec = Img2Vec(cuda=False)
    list_of_pil_images, filenames = loader.get_PIL_images()
    img_emb = img2vec.get_vec(list_of_pil_images)

    df_img = pd.DataFrame([filenames, img_emb]).T
    df_img = df_img.rename(columns={0: 'MemeName', 1: 'MemeVector'})
    df_img.to_csv('data/img_vec.csv', sep='|', index=False)

if prepare_txt_vec:
    doc2vec = Doc2Vec.load('data/doc2vec_wiki_d300_n5_w8_mc50_t12_e10_dbow.model')
    print('doc2vec loaded! \n')
    list_of_words = loader.get_memes_dataframe()
    txt_emb = [doc2vec.infer_vector(words) for words in list_of_words["CaptionText"]]

    df_txt = pd.DataFrame(
        list(zip(list_of_words["MemeLabel"],
                 list_of_words["CaptionText"],
                 list_of_words['HashId'],
                 txt_emb)),
        columns=['MemeLabel', 'CaptionText', 'HashId', 'TextVector'])
    df_txt.to_csv('data/txt_vec.csv', sep='|', index=False)
