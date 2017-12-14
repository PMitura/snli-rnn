"""
Trainer module. Responsible for training the neural network, and saving the created model.
"""

import json
import numpy as np
import os
import random
import sys
import rnn.logger as logger
import rnn.preprocessor as prep
import tensorflow as tf

RNN_HIDDEN_COUNT = 300
EPOCH_COUNT = 20
LEARNING_RATE = 0.003
BATCH_SIZE = 64
BATCH_CEILING = 5000
NUM_CLASSES = 3


# Loads embedding matrix as a tensorflow variable
def load_embedding_matrix():
    if not os.path.exists(prep.PRECOMPUTED_EMB_MATRIX_PATH):
        logger.error("Embedding matrix not found, please run preprocessor first.")
        sys.exit(1)
    with open(prep.PRECOMPUTED_EMB_MATRIX_PATH, 'r') as matrix_file:
        matrix = json.load(matrix_file)
    np_matrix = np.array(matrix, np.float32)

    # whether to keep this trainable or not is up to discussion
    tf_matrix = tf.get_variable(name="tf_matrix", shape=np_matrix.shape,
                                initializer=tf.constant_initializer(np_matrix), trainable=False)
    return tf_matrix


def load_padded_matrix(path):
    with open(path, 'r') as file:
        rows = json.load(file)
    maxlen = 0
    for row in rows:
        maxlen = max(maxlen, len(row))
    matrix = np.zeros((len(rows), maxlen), dtype=np.int32)
    for idx, row in enumerate(rows):
        matrix[idx] = np.pad(row, (0, maxlen - len(row)), mode='constant')
    return matrix


def labels_to_onehot(label_ids):
    labels = np.zeros((len(label_ids), NUM_CLASSES), np.bool)
    for idx, lid in enumerate(label_ids):
        labels[idx][lid] = 1
    return labels


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
    labels = labels_to_onehot(label_ids)
    return premise, hypothesis, labels


def load_test_matrices():
    if not os.path.exists(prep.PRECOMPUTED_TEST_HYPOTHESES_PATH) \
            or not os.path.exists(prep.PRECOMPUTED_TEST_PREMISES_PATH) \
            or not os.path.exists(prep.PRECOMPUTED_TEST_LABELS_PATH):
        logger.error("Testing matrices not found, please run preprocessor first.")
        sys.exit(1)
    premise = load_padded_matrix(prep.PRECOMPUTED_TEST_PREMISES_PATH)
    hypothesis = load_padded_matrix(prep.PRECOMPUTED_TEST_HYPOTHESES_PATH)
    with open(prep.PRECOMPUTED_TEST_LABELS_PATH, 'r') as labels_file:
        label_ids = json.load(labels_file)
    labels = labels_to_onehot(label_ids)
    return premise, hypothesis, labels


def produce_batch(premise, hypothesis, labels, queue, batch_size=BATCH_SIZE):
    num_steps_premise = premise.shape[1]
    num_steps_hypothesis = hypothesis.shape[1]

    i = queue.dequeue()
    premise_batch = tf.slice(premise, [i * batch_size, 0], [batch_size, num_steps_premise])
    hypothesis_batch = tf.slice(hypothesis, [i * batch_size, 0], [batch_size, num_steps_hypothesis])
    labels_batch = tf.slice(labels, [i * batch_size, 0], [batch_size, labels.shape[1]])

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


# get error average for given prediction
def get_error(prediction, labels):
    mistakes = tf.not_equal(tf.argmax(labels, 1), tf.argmax(prediction, 1))
    return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def time_distributed_dense(in_mx):
    in_reshaped = tf.reshape(in_mx, [-1, in_mx.get_shape()[2]])
    out_mx = tf.layers.dense(in_reshaped, RNN_HIDDEN_COUNT, activation=tf.nn.relu)
    out_mx = tf.reshape(out_mx, [-1, tf.shape(in_mx)[1], in_mx.get_shape()[2]])
    return out_mx


def build_model(premise_batch, hypothesis_batch, label_batch, embedding_matrix):
    # vectors with individual sequence lengths
    num_steps_premise = tf.shape(premise_batch)[1]
    num_steps_hypothesis = tf.shape(hypothesis_batch)[1]

    # build separate RNNs for premise and hypothesis
    with tf.variable_scope("premise_network"):
        premise_embeddings = tf.nn.embedding_lookup(embedding_matrix, premise_batch)
        premise_lengths = get_sequence_lengths(premise_embeddings)
        premise_shifted = time_distributed_dense(premise_embeddings)
        gru_premise_layer = tf.nn.rnn_cell.GRUCell(RNN_HIDDEN_COUNT)
        output_premise, states_premise = tf.nn.dynamic_rnn(
            cell=gru_premise_layer,
            inputs=premise_shifted,
            dtype=tf.float32,
            sequence_length=premise_lengths
        )

    with tf.variable_scope("hypothesis_network"):
        hypothesis_embeddings = tf.nn.embedding_lookup(embedding_matrix, hypothesis_batch)
        hypothesis_lengths = get_sequence_lengths(hypothesis_embeddings)
        hypothesis_shifted = time_distributed_dense(hypothesis_embeddings)
        gru_hypothesis_layer = tf.nn.rnn_cell.GRUCell(RNN_HIDDEN_COUNT)
        output_hypothesis, states_hypothesis = tf.nn.dynamic_rnn(
            cell=gru_hypothesis_layer,
            inputs=hypothesis_shifted,
            dtype=tf.float32,
            sequence_length=hypothesis_lengths
        )

    # get the last elements of RNN output matching the length of the sequence, without padding
    premise_last = last_relevant(output_premise, premise_lengths, num_steps_premise)
    hypothesis_last = last_relevant(output_hypothesis, hypothesis_lengths, num_steps_hypothesis)

    # merge networks, apply dense layers
    rnn_join = tf.concat([premise_last, hypothesis_last], 1)
    # drop_join = tf.nn.dropout(rnn_join, 0.5)
    dense1 = tf.layers.dense(rnn_join, RNN_HIDDEN_COUNT * 2, activation=tf.nn.relu)
    # drop1 = tf.nn.dropout(dense1, 0.5)
    dense2 = tf.layers.dense(dense1, RNN_HIDDEN_COUNT * 2, activation=tf.nn.relu)
    # drop2 = tf.nn.dropout(dense2, 0.5)
    dense3 = tf.layers.dense(dense2, RNN_HIDDEN_COUNT * 2, activation=tf.nn.relu)

    # softmax classification layer on output
    num_classes = NUM_CLASSES
    weight = tf.Variable(tf.truncated_normal([RNN_HIDDEN_COUNT * 2, num_classes], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    prediction = tf.nn.softmax(tf.matmul(dense3, weight) + bias)

    # feed results into optimizer
    offset = tf.constant(1e-8, shape=[num_classes])  # prevent NaN loss in case of log(0)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(tf.add(prediction, offset)), [1]))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    error = get_error(prediction, label_batch)
    return optimizer.minimize(cross_entropy), cross_entropy, error


def run():
    logger.header("Running trainer module.")

    logger.info("Loading embedding matrix into tensorflow model.")
    embedding_matrix = load_embedding_matrix()
    logger.success("Matrix loaded.")

    logger.info("Loading training data matrices.")
    train_premise_matrix, train_hypothesis_matrix, train_labels = load_train_matrices()
    test_premise_matrix, test_hypothesis_matrix, test_labels = load_test_matrices()
    logger.success("Matrices loaded.")

    logger.info("Building Tensorflow model.")

    # Placeholders for the feed dict
    premise_ph = tf.placeholder(tf.int32, [None, None])
    hypothesis_ph = tf.placeholder(tf.int32, [None, None])
    labels_ph = tf.placeholder(tf.float32, [None, train_labels.shape[1]])

    # Model is given as optimizer minimize operation
    model, loss, error = build_model(premise_ph, hypothesis_ph, labels_ph, embedding_matrix)

    # create batch producers for both training and testing
    num_batches = min(BATCH_CEILING, train_labels.shape[0] // BATCH_SIZE)
    num_test_batches = min(BATCH_CEILING, test_labels.shape[0] // BATCH_SIZE)
    train_batch_queue = tf.train.range_input_producer(limit=num_batches, shuffle=True)
    test_batch_queue = tf.train.range_input_producer(limit=num_test_batches, shuffle=False)
    premise_tf, hypothesis_tf, label_tf = produce_batch(train_premise_matrix, train_hypothesis_matrix, train_labels,
                                                        train_batch_queue)
    premise_ts, hypothesis_ts, label_ts = produce_batch(test_premise_matrix, test_hypothesis_matrix, test_labels,
                                                        test_batch_queue)
    logger.success("Model built.")

    logger.info("Running Tensorflow session. Good luck.")
    with tf.Session() as session:
        # Wouldn't work without this, for some reason
        input_coord = tf.train.Coordinator()
        input_threads = tf.train.start_queue_runners(session, coord=input_coord)

        session.run(tf.global_variables_initializer())

        for epoch in range(1, EPOCH_COUNT + 1):
            logger.info("Epoch " + str(epoch) + " startup...", level=2)

            # Run training on all batches (optimizer on)
            sum_loss = 0
            sum_err = 0
            for batch in range(1, num_batches + 1):
                premise_batch, hypothesis_batch, labels_batch = session.run([premise_tf, hypothesis_tf, label_tf])
                _, curr_loss, curr_err = session.run([model, loss, error],
                                                     {premise_ph: premise_batch,
                                                      hypothesis_ph: hypothesis_batch,
                                                      labels_ph: labels_batch})
                sum_loss += curr_loss
                sum_err += curr_err
                if batch % 100 == 0 and batch > 0:
                    logger.info("Batch " + str(batch) + ", loss: " + str(sum_loss / batch)
                                + "    acc.: " + str((1 - sum_err / batch) * 100), level=3)

            # Run testing on all batches (optimizer off)
            test_loss = 0
            test_err = 0
            for test_batch in range(1, num_test_batches + 1):
                premise_batch_t, hypothesis_batch_t, labels_batch_t = session.run([premise_ts, hypothesis_ts, label_ts])
                curr_loss, curr_err = session.run([loss, error],
                                                  {premise_ph: premise_batch_t,
                                                   hypothesis_ph: hypothesis_batch_t,
                                                   labels_ph: labels_batch_t})
                test_loss += curr_loss
                test_err += curr_err
            test_loss /= num_test_batches
            test_err /= num_test_batches
            logger.info("Epoch " + str(epoch) + " done. Test loss: " + str(test_loss)
                        + "    Test acc: " + str((1 - test_err) * 100), level=2)

        input_coord.request_stop()
        input_coord.join(input_threads)
    logger.success("Session run complete")
