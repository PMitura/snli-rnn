"""
Preprocessor module. Responsible for transforming the dataset JSONs into input arrays feedable to the neural network.
"""

import itertools
import json_lines
import os
import rnn.logger as logger
from rnn.downloader import check_all_unpacked, unpacked_dataset_path, unpacked_glove_path


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
    bar = logger.get_progress_bar("Reading GloVe vectors", max=20)
    total_size = os.path.getsize(filename)
    found_counter = 0;
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
    logger.info("Found vectors for " + str(found_counter) + " words out of " + str(len(word_filter)) + "."\
                + " Elapsed time: " + str(bar.elapsed) + " s")
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


def run():
    logger.header("Running preprocessor module.")

    if not check_all_unpacked():
        logger.error("Unpacked datasets or word vectors are missing. Please run downloader prior to preprocessor.")

    logger.info("Loading datasets into memory")
    try:
        train_dataset = json_to_array(unpacked_dataset_path() + "/snli_1.0_train.jsonl")
        test_dataset = json_to_array(unpacked_dataset_path() + "/snli_1.0_test.jsonl")
    except FileNotFoundError as error:
        logger.error("File: " + error.filename + " not found")
        return
    logger.success("Datasets loaded.")

    logger.info("Loading word vectors into memory")
    # Get a set of words used in datasets, so we don't store useless word vectors.
    wordset = set()
    get_used_words(train_dataset, wordset)
    get_used_words(test_dataset, wordset)
    # Load needed part of word vectors. Might induce large memory costs.
    try:
        word_vectors = wordvec_to_dict(unpacked_glove_path() + "/glove.42B.300d.txt", wordset)
    except FileNotFoundError as error:
        logger.error("File: " + error.filename + " not found")
        return
    logger.success("Word vectors loaded.")

