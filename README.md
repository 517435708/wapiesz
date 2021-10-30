# config

- working on latest Anaconda3 version: https://www.anaconda.com/products/individual
- CUDA: `conda install -c anaconda cudatoolkit=11.3`
- cudnn `conda install -c anaconda cudnn=8.2.1=cuda11.3_0`
---

## important libraries

`tensorflow==2.4.0`

## Datasets
[TODO]

 - [ ] find better datasets!!!!
 - [x] before start run [unzip-from-link.sh](unzip-from-link.sh) or [download-dataset.py](download-dataset.py)

#### example unzip-from-link.sh usage:

`bash unzip-from-link.sh "www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip" "./datasets/"`

#### example of download-dataset.py usage:

`python download-dataset.py -u https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip -e ./datasets/
`

---


## progress
tested on GPUs:

- Nvidia GTX 1050ti (mobile)
- ...

---


