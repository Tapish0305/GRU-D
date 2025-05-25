import tensorflow as tf

def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return tf.nn.dropout(ones, rate=rate)
    if count == 1:
        return tf.keras.backend.in_train_phase(
            dropped_inputs,
            ones,
            training=training
        )
    else:
        return [tf.keras.backend.in_train_phase(
            dropped_inputs,
            ones,
            training=training
        ) for _ in range(count)]
