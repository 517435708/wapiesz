import ai
import tensorflow as tf

ai.utils.device.allow_memory_growth()
#TODO after loading data
#
# train_df, val_df = ai.datasets.basic.prepare_train_test(128)
# model = ai.models.image.classifier.ImageClassifier(10)
# optimizer = tf.keras.optimizers.Adam(0.001)
#
# for images, classes in train_df:
#     images = images[:, :, :, tf.newaxis]
#     images = tf.cast(images, tf.float32) / 255.0
#
#     with tf.GradientTape() as tape:
#         outputs = model(images, training=False)
#         loss = ai.losses.classification.classification_loss(classes, outputs)
#         loss = tf.reduce_mean(loss)
#
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
# for images, classes in val_df:
#     print('test')
#     break