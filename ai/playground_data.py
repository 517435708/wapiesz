import ai
import tensorflow as tf
import numpy as np
import ai.datasets.loader as loader
import ai.models as models

ai.utils.device.allow_memory_growth()
# TODO prepare model!

verbose = True

df = loader.get_memes_dataframe()
X_train, y_train, X_val, y_val, X_test, y_test = loader.train_validate_test_split(df, label_column='title',
                                                                                  train_percent=.6, validate_percent=.4)

if verbose:
    print('\n\n\n')
    print(X_train.describe)
    print(X_val.describe)
    print(X_test.describe)

model = models.multimodal.text_meme.TM_Model()