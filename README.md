# config

- working on latest Anaconda3 version: https://www.anaconda.com/products/individual
- CUDA: `conda install -c anaconda cudatoolkit=11.3`
- cudnn `conda install -c anaconda cudnn=8.2.1=cuda11.3_0`
---

# How to run files in `memenet`

```
# 1. install PyTorch however you want

# 2. install requirements
pip install -r requirements.txt

(if you need doc2vec model)
# 3. download model from https://github.com/kongyq/Pretrained_Wikipedia_Doc2Vec_Models 
and add it to memenet/data 
and go to: Datasets section
NOTE: prepared vectors: https://drive.google.com/file/d/1OiW03hEBjwM5ODuy3a3un2TySZQp_DXl/view?usp=sharing

# 4. train imgnet and txtnet
# run this cmd in the same directory as README.md
python -m memenet.train

# 5. run streamlit app
streamlit run memenet/app.py
```
## important libraries

`tensorflow==2.4.0`
<a name="custom_anchor_name"></a>
## Datasets
Run [playground_dataset](ai/playground_datasets.py) if any problem occurs try:
```
1.pip install kaggle
2.cd ~/.kaggle
3.homepage www.kaggle.com -> Your Account -> Create New API token
4.mv ~/Downloads/kaggle.json ./
5.chmod 600 ./kaggle.json 
```
Then run [make_dataset](memenet/make_dataset.py), so you get vec_txt and vec_img csv files

#### how to use

## progress
tested on GPUs:

- Nvidia GTX 1050ti (mobile)
- ...

---

### jobs

[BlaiseCz](https://github.com/BlaiseCz):
- [x] prepare project architecture
- [x] prepare datasets
- [ ] complete README.md




