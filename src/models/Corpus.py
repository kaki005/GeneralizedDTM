import enum
from asyncio import proactor_events
from tkinter import SE

import numpy as np
import tensorflow as tf
from pandas import DataFrame, DatetimeIndex, Series, Timestamp


class Corpus:
    def __init__(self, df: DataFrame, vocabulary: Series, timestamps: DatetimeIndex):
        # self.documents: tf.Tensor = tf.zeros(1)
        self.df: DataFrame = df
        self.N: int = df.shape[0]
        self.vocabulary: Series = vocabulary
        self.timestamps: DatetimeIndex = timestamps
        self.unique_times: DatetimeIndex = self.timestamps.unique()
        self.time2index = {}
        """convert timestamp to index"""
        self.word2index = {}
        """(key: string, value: word_index)"""
        self.doc2time = {}
        """dictionary (key: doc_index, value: time_index)"""
        self.doc_words_and_counts: list[tuple[list[int], tf.Tensor]] = []
        """(document, (word_index, counts))"""
        self.document_indexes_per_time = []
        for i, (word, count) in enumerate(vocabulary.items()):
            self.word2index[word] = i
        for i, time in enumerate(self.unique_times):
            self.time2index[time.to_datetime64()] = i
            self.document_indexes_per_time.append([])
        for doc_idx, time in enumerate(timestamps):
            time_index = self.time2index[time.to_datetime64()]
            self.doc2time[doc_idx] = time_index
            self.document_indexes_per_time[time_index].append(doc_idx)
        for doc_name, doc_series in df.items():
            word_indexes = []
            counts = []
            for word, count in doc_series[doc_series > 0].items():
                word_indexes.append(self.word2index[word])
                counts.append(count)
            self.doc_words_and_counts.append((word_indexes, tf.convert_to_tensor(counts, dtype=tf.float64)))

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

    def get_words_and_counts(self, doc_index: int) -> tuple[list[int], tf.Tensor]:
        return self.doc_words_and_counts[doc_index]

    def get_unique_timestamps(self) -> DatetimeIndex:
        return self.unique_times
