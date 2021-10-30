import tensorflow as tf
import ai

class ImageClassifier(tf.keras.Model):

    def __init__(self, num_classes):
        super().__init__(name='image_classifier')

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 3, 2, 'same', activation='relu'),
            tf.keras.layers.Conv2D(16, 3, 2, 'same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, 2, 'same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='relu')
        ])

        self.optimizer = tf.keras.optimizers.Adam(0.001)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def query(self, images, classes, training):
        outputs = self.model(images, training=training)
        loss = ai.losses.classification.classification_loss(classes, outputs)
        return tf.reduce_mean(loss)

    def train(self, images, classes):
        with tf.GradientTape() as tape:
            loss = self.query(images, classes, True)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))