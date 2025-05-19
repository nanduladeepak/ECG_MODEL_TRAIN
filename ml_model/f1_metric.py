import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert one-hot to label
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)

        # compute booleans
        true_pos = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(y_true, y_pred),
                                   tf.not_equal(y_true, 0)), self.dtype)
        )
        false_pos = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_true, y_pred),
                                   tf.equal(y_pred, 0)), self.dtype)
        )
        false_neg = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_true, y_pred),
                                   tf.equal(y_true, 0)), self.dtype)
        )

        self.tp.assign_add(true_pos)
        self.fp.assign_add(false_pos)
        self.fn.assign_add(false_neg)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall    = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)
