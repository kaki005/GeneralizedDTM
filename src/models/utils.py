import numpy as np
import tensorflow as tf


def log_add(x, y):
    return tf.math.log(tf.math.exp(x) + tf.math.exp(y))
