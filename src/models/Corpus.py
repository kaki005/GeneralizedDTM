import enum
from asyncio import proactor_events
from tkinter import SE

import numpy as np
import tensorflow as tf
from pandas import DataFrame, Series, Timestamp


class Corpus:
    def __init__(self, df: DataFrame, vocabulary: Series, timestamps: list[Timestamp]):
        # self.documents: tf.Tensor = tf.zeros(1)
        self.df: DataFrame = df
        self.N: int = df.shape[0]
        self.vocabulary: Series = vocabulary
        self.timestamps: list[Timestamp] = timestamps
        self.unique_times: np.ndarray = np.unique(self.timestamps)
        self.time2index = {}
        """convert timestamp to index"""
        self.word2index = {}
        """(key: string, value: word_index)"""
        self.doc2time = {}
        """dictionary (key: doc_index, value: time_index)"""
        self.doc_words_and_counts: list[tuple[list[int], list[int]]] = []
        """(document, (word_index, counts))"""
        self.document_indexes_per_time = []
        for i, word in enumerate(vocabulary):
            self.word2index[word] = i
        for i, time in enumerate(self.unique_times):
            self.time2index[time] = i
            self.document_indexes_per_time.append([])
        for doc_idx, time in enumerate(timestamps):
            self.doc2time[doc_idx] = self.time2index[time]
            self.document_indexes_per_time[self.time2index[time]].append(doc_idx)
        for doc_name, doc_series in df.items():
            word_indexes = []
            counts = []
            for word, count in doc_series[doc_series > 0].items():
                word_indexes.append(self.word2index[word])
                counts.append(count)
            self.doc_words_and_counts.append((word_indexes, counts))

    @property
    def lexycon(self) -> Series:
        return self.vocabulary

    @property
    def T(self) -> int:
        return len(self.unique_times)

    def get_timestamp_index(self, doc_index: int) -> int:
        return self.doc2time[doc_index]

    def get_datapoints_for_timestamp(self, time_index: int):
        return self.document_indexes_per_time[time_index]

    def get_words_and_counts(self, doc_index: int) -> tuple[list[int], list[int]]:
        return self.doc_words_and_counts[doc_index]

    def get_unique_timestamps(self) -> np.ndarray:
        return self.unique_times
