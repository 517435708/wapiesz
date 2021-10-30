import tensorflow as tf
from sklearn.model_selection import train_test_split


def prepare_train_test(X, y, batch_size, val_set=False):
    val_ds = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    if val_set:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                          random_state=1)  # 0.25 x 0.8 = 0.2
        val_ds = tf.data.Dataset.from_tensor_slices(X_val) \
            .shuffle(buffer_size=1024) \
            .batch(batch_size=batch_size) \
            .prefetch(8)

    train_ds = tf.data.Dataset.from_tensor_slices(X_train) \
        .shuffle(buffer_size=1024) \
        .batch(batch_size=batch_size) \
        .prefetch(8)

    test_ds = tf.data.Dataset.from_tensor_slices(X_test) \
        .shuffle(buffer_size=1024) \
        .batch(batch_size=batch_size) \
        .prefetch(8)

    return train_ds, test_ds, val_ds
