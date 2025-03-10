import numpy as np
import tensorflow as tf


def log_add(x, y):
    return tf.math.log(tf.math.exp(x) + tf.math.exp(y))


def randn(maxval: int):
    tf.random.uniform(shape=(), minval=0, maxval=maxval, dtype=tf.int32)
