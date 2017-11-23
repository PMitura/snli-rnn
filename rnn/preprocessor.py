"""
Preprocessor module. Responsible for transforming the dataset JSONs into input arrays feedable to the neural network.
"""

import itertools
import json_lines
import rnn.logger as logger
from rnn.downloader import is_unpacked, unpacked_dataset_path


# Converts json file to array of python dictionaries
def json_to_array(filename, limit=100000):
    lines = []
    with open(filename, "rb") as jsonl_file:
        for line in itertools.islice(json_lines.reader(jsonl_file), 0, limit):
            lines.append(line)
    return lines


def run():
    logger.header("Running preprocessor module.")

    if not is_unpacked():
        logger.error("No unpacked dataset found. Please run downloader prior to preprocessor.")

    logger.info("Loading JSON train and test datasets")
    try:
        train_dataset = json_to_array(unpacked_dataset_path() + "/snli_1.0_train.jsonl")
        test_dataset = json_to_array(unpacked_dataset_path() + "/snli_1.0_test.jsonl")
    except FileNotFoundError as error:
        logger.error("File: " + error.filename + " not found")
        return
    logger.success("Dataset files loaded.")

