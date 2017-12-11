"""
Trainer module. Responsible for training the neural network, and saving the created model.
"""

import json
import numpy as np
import os
import sys
import rnn.logger as logger
import rnn.preprocessor as prep
import tensorflow as tf


# Loads embedding matrix as a tensorflow variable
def load_embedding_matrix():
    if not os.path.exists(prep.PRECOMPUTED_EMB_MATRIX_PATH):
        logger.error("Embedding matrix not found, please run preprocessor first.")
        sys.exit(1)
    with open(prep.PRECOMPUTED_EMB_MATRIX_PATH, 'r') as matrix_file:
        matrix = json.load(matrix_file)
    np_matrix = np.array(matrix, np.float32)
    tf_matrix = tf.get_variable(name="tf_matrix", shape=np_matrix.shape,
                                initializer=tf.constant_initializer(np_matrix), trainable=False)
    return tf_matrix


def load_padded_matrix(path):
    with open(path, 'r') as file:
        rows = json.load(file)
    maxlen = 0
    for row in rows:
        maxlen = max(maxlen, len(row))
    matrix = np.zeros((len(rows), maxlen))
    for idx, row in enumerate(rows):
        matrix[idx] = np.pad(row, (0, maxlen - len(row)), mode='constant')
    return matrix


def load_train_matrices():
    if not os.path.exists(prep.PRECOMPUTED_TRAIN_HYPOTHESES_PATH) \
            or not os.path.exists(prep.PRECOMPUTED_TRAIN_PREMISES_PATH) \
            or not os.path.exists(prep.PRECOMPUTED_TRAIN_LABELS_PATH):
        logger.error("Training matrices not found, please run preprocessor first.")
        sys.exit(1)
    premise = load_padded_matrix(prep.PRECOMPUTED_TRAIN_PREMISES_PATH)
    hypothesis = load_padded_matrix(prep.PRECOMPUTED_TRAIN_HYPOTHESES_PATH)
    with open(prep.PRECOMPUTED_TRAIN_LABELS_PATH, 'r') as labels_file:
        labels = json.load(labels_file)
    return premise, hypothesis, labels


def produce_batch(premise, hypothesis, labels, batch_size=32, name=None):
    with tf.name_scope(name, "producers", [premise, hypothesis, labels, batch_size]):
        premise = tf.convert_to_tensor(premise, name="premise", dtype=tf.int32)
        hypothesis = tf.convert_to_tensor(hypothesis, name="hypothesis", dtype=tf.int32)
        labels = tf.convert_to_tensor(labels, name="labels", dtype=tf.int32)

        input_len = tf.shape(premise)[0]
        num_steps = tf.shape(premise)[1]
        batch_len = input_len // batch_size

        i = tf.train.range_input_producer(limit=batch_len, shuffle=False).dequeue()
        premise_batch = tf.slice(premise, [i * batch_size, 0], [batch_size, num_steps], name="premise_batch")
        hypothesis_batch = tf.slice(hypothesis, [i * batch_size, 0], [batch_size, num_steps], name="hypotheis_batch")
        labels_batch = tf.slice(labels, [i * batch_size], [batch_size], name="labels_batch")

        return premise_batch, hypothesis_batch, labels_batch


def run():
    logger.header("Running trainer module.")

    logger.info("Loading embedding matrix into tensorflow model")
    embedding_matrix = load_embedding_matrix()
    logger.success("Matrix loaded")

    logger.info("Loading training data matrices")
    premise_matrix, hypothesis_matrix, labels = load_train_matrices()
    logger.success("Matrices loaded")

    logger.info("Building tensorflow model")
    premise_batch, hypothesis_batch, label_batch = produce_batch(premise_matrix, hypothesis_matrix, labels)
    word_embeddings = tf.nn.embedding_lookup(embedding_matrix, premise_batch)
    logger.success("Model built")

