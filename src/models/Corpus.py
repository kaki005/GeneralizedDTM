import tensorflow as tf
from pandas import Timestamp


class Corpus:
    def __init__(self):
        self.documents: tf.Tensor = tf.zeros(1)

    @property
    def lexycon(self) -> list[int]:
        return []

    @property
    def N(self) -> int:
        return 0

    def get_datapoints_for_timestamp(self):
        pass

    def get_words_and_counts(self) -> int:
        return 0

    def get_unique_timestamps(self) -> list[Timestamp]:
        return []
