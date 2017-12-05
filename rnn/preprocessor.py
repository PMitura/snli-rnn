"""
Preprocessor module. Responsible for transforming the dataset JSONs into input arrays feedable to the neural network.
"""

import itertools
import json
import json_lines
import numpy as np
import os
import rnn.logger as logger
import time
from rnn.downloader import check_all_unpacked, unpacked_dataset_path, unpacked_glove_path

PRECOMPUTED_GLOVE_PATH = "data/glove/word_vectors.json"
PRECOMPUTED_MATRIX_PATH = "data/glove/embedding_matrix.json"
PRECOMPUTED_TRAIN_DATA_PATH = "data/train_data_matrix.json"
PRECOMPUTED_TEST_DATA_PATH = "data/test_data_matrix.json"


# Converts json file to array of python dictionaries
def json_to_array(filename, limit=100000):
    lines = []
    with open(filename, "rb") as jsonl_file:
        for line in itertools.islice(json_lines.reader(jsonl_file), 0, limit):
            lines.append(line)
    return lines


# Transforms GloVe data file into dictionary of word vectors
def wordvec_to_dict(filename, word_filter):
    vec_dict = {}

    # progress bar overhead
    bar = logger.get_progress_bar("Reading GloVe vectors", level=2, limit=20)
    total_size = os.path.getsize(filename)
    found_counter = 0
    size_counter = 0
    last_milestone = 0
    with open(filename, "r") as file:
        for raw_line in file:
            # progress bar overhead
            size_counter += len(raw_line) + 1
            if int((size_counter / total_size) * 20) > last_milestone:
                bar.next()
                last_milestone += 1

            # process line
            line = raw_line.split()
            if not line[0] in word_filter:
                continue
            found_counter += 1
            num_line = [float(x) for x in line[1:]]
            vec_dict[line[0]] = num_line

    bar.finish()
    # Observation: most of the unmatched words are typos, compounds or really uncommon.
    logger.info("Found vectors for " + str(found_counter) + " words out of " + str(len(word_filter)) + "."
                + " Elapsed time: " + str(bar.elapsed) + " s", level=2)
    return vec_dict


# Converts sentence to a list of lowercase words without special characters.
def sentence_to_words(sentence):
    sentence = sentence.lower()
    sentence = ''.join([i for i in sentence if i.isalnum() or i.isspace()])
    return sentence.split()


# Extracts a set of words used in sentences of a given dataset
def get_used_words(dataset, wordset=set()):
    for sentence_pair in dataset:
        for item in sentence_to_words(sentence_pair["sentence1"]):
            wordset.add(item)
        for item in sentence_to_words(sentence_pair["sentence2"]):
            wordset.add(item)
    return wordset


# Assigns ID to every key in a dictionary
def generate_dictionary_ids(src_dict):
    counter = 0
    id_dict = {}
    for key in src_dict:
        id_dict[key] = counter
        counter += 1
    return id_dict


# Transforms string vocabulary and embeddings dictionary into ID indexed embedding matrix
def generate_embedding_matrix(embeddings_dict, word_id_mapping):
    embedding_matrix = np.zeros(shape=(len(embeddings_dict), 300))
    for key, item in embeddings_dict.items():
        embedding_matrix[word_id_mapping[key]] = np.array(item)
    return embedding_matrix


# Translates list of sentence pairs into two matrices of their word IDs.
# Words in sentences are skipped, if no matching word vector is provided (= missing ID in mappings)
def input_data_to_matrices(dataset, word_id_mapping):
    premise_matrix = []
    hypothesis_matrix = []
    # sentence1 denotes premise, sentence2 is hypothesis
    for sentence_pair in dataset:
        premise_row = []
        hypothesis_row = []
        for item in sentence_to_words(sentence_pair["sentence1"]):
            if word_id_mapping.count(item):
                premise_row.append(word_id_mapping[item])
        for item in sentence_to_words(sentence_pair["sentence2"]):
            if word_id_mapping.count(item):
                hypothesis_row.append(word_id_mapping[item])
        premise_matrix.append(premise_row)
        hypothesis_matrix.append(hypothesis_row)
    return premise_matrix, hypothesis_matrix


# Run all preprocessing routines
def run(force_recompute=False):
    logger.header("Running preprocessor module.")

    if not check_all_unpacked():
        logger.error("Unpacked datasets or word vectors are missing. Please run downloader prior to preprocessor.")

    time_start = time.time()
    logger.info("Loading datasets into memory")
    try:
        train_dataset = json_to_array(unpacked_dataset_path() + "/snli_1.0_train.jsonl")
        test_dataset = json_to_array(unpacked_dataset_path() + "/snli_1.0_test.jsonl")
    except FileNotFoundError as error:
        logger.error("File: " + error.filename + " not found")
        return
    time_end = time.time()
    logger.success("Datasets loaded. Elapsed time: " + "{0:.2f}".format(time_end - time_start) + " s")

    embeddings_changed = False
    time_start = time.time()
    if os.path.exists(PRECOMPUTED_GLOVE_PATH) and not force_recompute:
        logger.info("Precomputed word vectors found, loading into memory.")
        with open(PRECOMPUTED_GLOVE_PATH, 'r') as infile:
            word_vectors = json.load(infile)
    else:
        logger.info("Loading word vectors into memory")
        # Get a set of words used in datasets, so we don't store useless word vectors.
        vocabulary = set()
        get_used_words(train_dataset, vocabulary)
        get_used_words(test_dataset, vocabulary)
        # Load needed part of word vectors. Might induce large memory costs.
        try:
            word_vectors = wordvec_to_dict(unpacked_glove_path() + "/glove.42B.300d.txt", vocabulary)
        except FileNotFoundError as error:
            logger.error("File: " + error.filename + " not found")
            return
        logger.info("Storing loaded vectors for future use.", level=2)
        with open(PRECOMPUTED_GLOVE_PATH, 'w') as outfile:
            json.dump(word_vectors, outfile)
        embeddings_changed = True
    time_end = time.time()
    logger.success("Word vectors loaded. Elapsed time: " + "{0:.2f}".format(time_end - time_start) + " s")

    if not os.path.exists(PRECOMPUTED_MATRIX_PATH) or force_recompute or embeddings_changed:
        logger.info("Generating initial embedding matrix.")
        id_mapping = generate_dictionary_ids(word_vectors)
        embedding_matrix = generate_embedding_matrix(word_vectors, id_mapping)
        logger.info("Storing embedding matrix for future use.", level=2)
        with open(PRECOMPUTED_MATRIX_PATH, 'w') as outfile:
            json.dump(embedding_matrix.tolist(), outfile)
        logger.success("Embedding matrix created.")
    else:
        logger.info("Embedding matrix found, skipping its computation.")

    logger.info("Creating train and test input matrices")
