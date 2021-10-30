import ai
import tensorflow as tf

ai.utils.device.allow_memory_growth()

images = tf.ones([16, 28, 28, 1], tf.float32)
classes = tf.random.uniform([16], 0, 10, dtype=tf.int32)

model = ai.models.image.classifier.ImageClassifier(10)
optimizer = tf.keras.optimizers.Adam(0.001)


@tf.function
def query(images, classes, training):
    outputs = model(images, training=training)
    loss = ai.losses.classification.classification_loss(classes, outputs)
    return tf.reduce_mean(loss)


@tf.function
def train(images, classes):
    with tf.GradientTape() as tape:
        loss = query(images, classes, True)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


query(images, classes, False)
train(images, classes)

import time

start = time.time()
for i in range(100):
    query(images, classes, False)
end = time.time()
print(f'time {end - start}')

start = time.time()
for i in range(100):
    train(images, classes)
end = time.time()
print(f'time {end - start}')
