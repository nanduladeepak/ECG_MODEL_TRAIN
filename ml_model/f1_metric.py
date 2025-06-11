import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_binary', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Convert probabilities to binary predictions
        y_pred_binary = tf.cast(y_pred >= 0.5, tf.int32)

        # Compute true positives, false positives, and false negatives
        true_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 1)), self.dtype))
        false_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 1)), self.dtype))
        false_neg = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 0)), self.dtype))

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            true_pos = tf.reduce_sum(true_pos * sample_weight)
            false_pos = tf.reduce_sum(false_pos * sample_weight)
            false_neg = tf.reduce_sum(false_neg * sample_weight)

        self.tp.assign_add(true_pos)
        self.fp.assign_add(false_pos)
        self.fn.assign_add(false_neg)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)
