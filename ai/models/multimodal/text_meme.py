import tensorflow as tf
import ai
# TODO prepare model!
#   - add loss function
class TM_Model(tf.keras.Model):

    def __init__(self, verbose=False):
        super().__init__(name='text_meme')

        self.model = tf.keras.Sequential([

        ])

        if verbose:
            self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(0.001)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    @tf.function
    def query(self, images, classes, training):
        # outputs = self.model(images, training=training)
        # loss = ai.losses.classification.classification_loss(classes, outputs)
        loss = tf.Tensor([1, 3, 4])
        return tf.reduce_mean(loss)

    @tf.function
    def train(self, images, classes):
        with tf.GradientTape() as tape:
            loss = self.query(images, classes, True)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
