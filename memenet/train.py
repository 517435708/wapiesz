import numpy as np
import pandas as pd
import torch as T
import torch.nn.functional as F
import torch.nn as nn

from math import ceil
from memenet.utils import pad_epoch, total_param_count
from memenet.models import ImgNet, TxtNet


def meme_loss(Y1, Y2):
    # return F.cosine_similarity(Y1, Y2, dim=1).mean()
    return ((Y1 - Y2) ** 2).sum(1).mean()  # euclidean whatever


def meme_train(*, img_net, txt_net, img_emb, txt_emb, epochs=50, batch_size=1024, lr=0.005):
    # TODO: early stopping, MAYBE lr decay if learning is too aggressive
    assert img_emb.shape[0] == txt_emb.shape[0]
    n_samples = img_emb.shape[0]
    n_batches = int(ceil(n_samples / batch_size))
    optim = T.optim.AdamW(list(img_net.parameters()) + list(txt_net.parameters()), lr=lr)
    for epoch in range(epochs):
        indices = np.random.choice(n_samples, n_samples, replace=False)
        epoch_loss = []
        for batch_idx in range(n_batches):
            batch = indices[batch_idx * batch_size:min((batch_idx + 1) * batch_size, n_samples)]
            x_img = img_emb[batch]
            x_txt = txt_emb[batch]
            Y_img = img_net(x_img)
            Y_txt = txt_net(x_txt)
            loss = meme_loss(Y_img, Y_txt)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(float(loss))
            print(
                f'\repoch {pad_epoch(epoch + 1, epochs)}/{epochs} | batch {pad_epoch(batch_idx + 1, n_batches)}/{n_batches} | batch loss {loss:.4f}',
                end='')
        print(f' | epoch loss {np.mean(epoch_loss):.4f}')


if __name__ == '__main__':
    from datetime import datetime

    data_path = 'data/'
    # load image and text embeddings
    # learn on random noise for now
    data_img = pd.read_pickle(data_path + 'img_vec.pkl')
    data_txt = pd.read_pickle(data_path + 'txt_vec.pkl')
    full_data = pd.merge(data_img, data_txt, on=['MemeLabel'], how="left", left_index=True)

    img_tr = T.tensor(full_data['MemeVector'])
    txt_tr = T.tensor(full_data['TextVector'])
    img_net = ImgNet(hidden_dim=128)
    txt_net = TxtNet()
    print('ImgNet param count:', total_param_count(img_net))
    print('TxtNet param count:', total_param_count(txt_net))

    meme_train(img_net=img_net, txt_net=txt_net, img_emb=img_tr, txt_emb=txt_tr, epochs=5)

    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    T.save(img_net, f'data/img_net-{timestamp}.pt')
    T.save(txt_net, f'data/txt_net-{timestamp}.pt')
