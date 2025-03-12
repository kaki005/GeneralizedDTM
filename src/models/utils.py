import numpy as np
import tensorflow as tf


def chol_inv(m):
    """
    Computes the inverse of a matrix using Cholesky decomposition if possible,
    otherwise falls back to standard matrix inversion.

    Args:
        m: A tensor representing a square matrix

    Returns:
        A tuple containing:
        - The inverse of the input matrix
        - The squared product of the diagonal elements of L_inv (or 1/det(m) as fallback)
    """
    try:
        # Make sure the matrix is Hermitian (symmetric in the real case)
        m_symm = (m + tf.transpose(m)) / 2
        # Compute the Cholesky decomposition
        L = tf.linalg.cholesky(m_symm)
        # Compute the inverse of L
        L_inv = tf.linalg.inv(tf.transpose(L))
        # Compute L_inv' * L_inv
        m_inv = tf.matmul(tf.transpose(L_inv), L_inv)

        # Compute the squared product of the diagonal elements of L_inv
        diag_L_inv = tf.linalg.diag_part(L_inv)
        det_factor = tf.square(tf.reduce_prod(diag_L_inv))

        return m_inv, det_factor

    except tf.errors.InvalidArgumentError:
        # Fallback: standard matrix inversion
        m_inv = tf.linalg.inv(m)

        # Compute 1/det(m)
        det_m = tf.linalg.det(m)
        det_factor = 1.0 / det_m

        return m_inv, det_factor


def log_add(x, y):
    return tf.math.log(tf.math.exp(x) + tf.math.exp(y))


def randn(maxval: int) -> int:
    return tf.random.uniform(shape=(), minval=0, maxval=maxval, dtype=tf.int32)
