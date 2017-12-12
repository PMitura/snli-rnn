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

RNN_HIDDEN_COUNT = 300
RNN_LAYERS_COUNT = 2


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


# gets vector of sequence lenghts, since sentences in our dataset have variable word counts
def get_sequence_lengths(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


# Returns last relevant output of the network, based on processed sequence length
def last_relevant(output, lengths):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (lengths - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def build_model(premise_matrix, hypothesis_matrix, labels, embedding_matrix):
    premise_batch, hypothesis_batch, label_batch = produce_batch(premise_matrix, hypothesis_matrix, labels)
    premise_embeddings = tf.nn.embedding_lookup(embedding_matrix, premise_batch)
    hypothesis_embeddings = tf.nn.embedding_lookup(embedding_matrix, hypothesis_batch)

    # vectors with individual sequence lengths
    premise_lengths = get_sequence_lengths(premise_embeddings)
    hypothesis_lengths = get_sequence_lengths(hypothesis_embeddings)

    num_steps = int(premise_matrix.shape[1])
    num_feats = int(premise_embeddings.get_shape()[2])
    print(num_steps)
    print(num_feats)
    premise_input = tf.placeholder(tf.float32, [None, num_steps, num_feats])
    hypothesis_input = tf.placeholder(tf.float32, [None, num_steps, num_feats])

    with tf.variable_scope("premise_network"):
        gru_premise_layers = [tf.nn.rnn_cell.GRUCell(RNN_HIDDEN_COUNT)] * RNN_LAYERS_COUNT
        multi_premise_cell = tf.nn.rnn_cell.MultiRNNCell(gru_premise_layers)
        output_premise, states_premise = tf.nn.dynamic_rnn(
            cell=multi_premise_cell,
            inputs=premise_input,
            dtype=tf.float32,
            sequence_length=premise_lengths
        )

    with tf.variable_scope("hypothesis_network"):
        gru_hypothesis_layers = [tf.nn.rnn_cell.GRUCell(RNN_HIDDEN_COUNT)] * RNN_LAYERS_COUNT
        multi_hypothesis_cell = tf.nn.rnn_cell.MultiRNNCell(gru_hypothesis_layers)
        output_hypothesis, states_hypothesis = tf.nn.dynamic_rnn(
            cell=multi_hypothesis_cell,
            inputs=hypothesis_input,
            dtype=tf.float32,
            sequence_length=hypothesis_lengths
        )

    premise_last = last_relevant(output_premise, premise_lengths)
    hypothesis_last = last_relevant(output_hypothesis, hypothesis_lengths)

    print(premise_last)


def run():
    logger.header("Running trainer module.")

    logger.info("Loading embedding matrix into tensorflow model.")
    embedding_matrix = load_embedding_matrix()
    logger.success("Matrix loaded.")

    logger.info("Loading training data matrices.")
    premise_matrix, hypothesis_matrix, labels = load_train_matrices()
    logger.success("Matrices loaded.")

    logger.info("Building Tensorflow model.")
    build_model(premise_matrix, hypothesis_matrix, labels, embedding_matrix)
    logger.success("Model built.")

    """
    res = tf.contrib.learn.run_n({"y": premise_lengths}, n=1, feed_dict=None)
    print("Batch shape: {}".format(res[0]["y"].shape))
    print(res[0]["y"])
    """
