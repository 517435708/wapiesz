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

# 3. train imgnet and txtnet
# run this cmd in the same directory as README.md
python -m memenet.train

# 4. run streamlit app
streamlit run memenet/app.py
```

## important libraries

`tensorflow==2.4.0`

## Datasets
[TODO]
We need to find better datasets!

currently working with https://www.kaggle.com/williamscott701/memotion-dataset-7k

#### how to use
https://www.kaggle.com/c/two-sigma-financial-news/discussion/83593

## progress
tested on GPUs:

- Nvidia GTX 1050ti (mobile)
- ...

---

### jobs

[BlaiseCz](https://github.com/BlaiseCz):
- [x] prepare project architecture
- [ ] prepare datasets
- [ ] complete README.md




