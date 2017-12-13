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

EPOCH_COUNT = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 32
BATCH_CEILING = 100


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
        label_ids = json.load(labels_file)

    # convert labels to one hot
    label_dict = {}
    label_counter = 0
    for lid in label_ids:
        if lid not in label_dict:
            label_dict[lid] = label_counter
            label_counter += 1
    labels = np.zeros((len(label_ids), label_counter))
    for idx, lid in enumerate(label_ids):
        labels[idx][label_dict[lid]] = 1
    return premise, hypothesis, labels


def produce_batch(premise, hypothesis, labels, batch_size=BATCH_SIZE, name=None):
    with tf.name_scope(name, "producers", [premise, hypothesis, labels, batch_size]):
        premise = tf.convert_to_tensor(premise, name="premise", dtype=tf.int32)
        hypothesis = tf.convert_to_tensor(hypothesis, name="hypothesis", dtype=tf.int32)
        labels = tf.convert_to_tensor(labels, name="labels", dtype=tf.float32)

        input_len = tf.shape(premise)[0]
        num_steps = tf.shape(premise)[1]
        ceiling = tf.constant(BATCH_CEILING, name="ceiling")
        batch_len = tf.minimum(ceiling, input_len // batch_size)

        i = tf.train.range_input_producer(limit=batch_len, shuffle=True).dequeue()
        premise_batch = tf.slice(premise, [i * batch_size, 0], [batch_size, num_steps], name="premise_batch")
        hypothesis_batch = tf.slice(hypothesis, [i * batch_size, 0], [batch_size, num_steps], name="hypotheis_batch")
        labels_batch = tf.slice(labels, [i * batch_size, 0], [batch_size, labels.shape[1]], name="labels_batch")

        return premise_batch, hypothesis_batch, labels_batch


# gets vector of sequence lenghts, since sentences in our dataset have variable word counts
def get_sequence_lengths(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


# Returns last relevant output of the network, based on processed sequence length
def last_relevant(output, lengths, max_length):
    batch_size = tf.shape(output)[0]
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

    # build separate RNNs for premise and hypothesis
    with tf.variable_scope("premise_network"):
        gru_premise_layers = [tf.nn.rnn_cell.GRUCell(size) for size in [RNN_HIDDEN_COUNT, RNN_HIDDEN_COUNT / 2]]
        multi_premise_cell = tf.nn.rnn_cell.MultiRNNCell(gru_premise_layers)
        output_premise, states_premise = tf.nn.dynamic_rnn(
            cell=multi_premise_cell,
            inputs=premise_embeddings,
            dtype=tf.float32,
            sequence_length=premise_lengths
        )

    with tf.variable_scope("hypothesis_network"):
        gru_hypothesis_layers = [tf.nn.rnn_cell.GRUCell(size) for size in [RNN_HIDDEN_COUNT, RNN_HIDDEN_COUNT / 2]]
        multi_hypothesis_cell = tf.nn.rnn_cell.MultiRNNCell(gru_hypothesis_layers)
        output_hypothesis, states_hypothesis = tf.nn.dynamic_rnn(
            cell=multi_hypothesis_cell,
            inputs=hypothesis_embeddings,
            dtype=tf.float32,
            sequence_length=hypothesis_lengths
        )

    # get the last elements of RNN output matching the length of the sequence, without padding
    premise_last = last_relevant(output_premise, premise_lengths, num_steps)
    hypothesis_last = last_relevant(output_hypothesis, hypothesis_lengths, num_steps)

    # merge networks into a single dense layer
    rnn_join = tf.concat([premise_last, hypothesis_last], 1)
    merged_nn = tf.layers.dense(rnn_join, RNN_HIDDEN_COUNT, activation=tf.nn.relu)

    # softmax classification layer on output
    num_classes = labels.shape[1]
    weight = tf.Variable(tf.truncated_normal([RNN_HIDDEN_COUNT, num_classes], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    prediction = tf.nn.softmax(tf.matmul(merged_nn, weight) + bias)

    # feed results into optimizer
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(prediction), [1]))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    return optimizer.minimize(cross_entropy), cross_entropy, prediction, label_batch


def run():
    logger.header("Running trainer module.")

    logger.info("Loading embedding matrix into tensorflow model.")
    embedding_matrix = load_embedding_matrix()
    logger.success("Matrix loaded.")

    logger.info("Loading training data matrices.")
    premise_matrix, hypothesis_matrix, labels = load_train_matrices()
    logger.success("Matrices loaded.")

    logger.info("Building Tensorflow model.")
    model, loss, p, l = build_model(premise_matrix, hypothesis_matrix, labels, embedding_matrix)
    logger.success("Model built.")

    logger.info("Running Tensorflow session. Good luck.")

    with tf.Session() as session:
        input_coord = tf.train.Coordinator()
        input_threads = tf.train.start_queue_runners(session, coord=input_coord)

        session.run(tf.global_variables_initializer())
        num_batches = min(BATCH_CEILING, len(labels) // BATCH_SIZE)
        for epoch in range(EPOCH_COUNT):
            logger.info("Epoch " + str(epoch) + " startup...", level=2)
            sum_loss = 0
            for batch in range(num_batches):
                _, curr_loss, pp, ll = session.run([model, loss, p, l])
                print(pp)
                print(ll)
                sum_loss += curr_loss
                logger.info("Batch " + str(batch) + " , curr. loss: " + str(curr_loss), level=3)
            logger.info("Epoch " + str(epoch) + " done. Avg loss: " + str(sum_loss / num_batches), level=2)

        input_coord.request_stop()
        input_coord.join(input_threads)
    logger.success("Session run complete")

    """ Copy for short runs
    res = tf.contrib.learn.run_n({"y": premise_lengths}, n=1, feed_dict=None)
    print("Batch shape: {}".format(res[0]["y"].shape))
    print(res[0]["y"])
    """
