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
    print(np_matrix.shape)
    tf_matrix = tf.get_variable(name="tf_matrix", shape=np_matrix.shape,
                                initializer=tf.constant_initializer(np_matrix), trainable=False)
    return tf_matrix


def load_train_matrices():
    if not os.path.exists(prep.PRECOMPUTED_TRAIN_HYPOTHESES_PATH)\
            or not os.path.exists(prep.PRECOMPUTED_TRAIN_PREMISES_PATH):
        logger.error("Training matrices not found, please run preprocessor first.")
        sys.exit(1)
    with open(prep.PRECOMPUTED_TRAIN_PREMISES_PATH, 'r') as premises_file:
        premise = json.load(premises_file)
    with open(prep.PRECOMPUTED_TRAIN_HYPOTHESES_PATH, 'r') as hypotheses_file:
        hypothesis = json.load(hypotheses_file)
    return premise, hypothesis


# def create_batch_producer(matrix, train_)


def run():
    logger.header("Running trainer module.")

    logger.info("Loading embedding matrix into tensorflow model")
    embedding_matrix = load_embedding_matrix()
    logger.info("Matrix loaded")

    logger.info("Loading training data matrix")
    train_premise_matrix, train_hypothesis_matrix = load_train_matrices()
    logger.info("Matrix loaded")

    logger.info("Building tensorflow model")
    # word_embeddings = tf.nn.embedding_lookup(embedding_matrix, )
    logger.info("Model built")

