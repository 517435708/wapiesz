import ai
import tensorflow as tf

ai.utils.device.allow_memory_growth()

images = tf.ones([16, 28, 28, 1], tf.float32)
classes = tf.random.uniform([16], 0, 10, dtype=tf.int32)

model = ai.models.image.classifier.ImageClassifier(10)
optimizer = tf.keras.optimizers.Adam(0.001)

with tf.GradientTape() as tape:
    outputs = model(images, training=False)
    loss = ai.losses.classification.classification_loss(classes, outputs)
    loss = tf.reduce_mean(loss)

grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print('aaa')
