"""
Downloader module, gets and unpacks the dataset if not present.
"""

import os
import wget
import zipfile
import rnn.logger as logger

# downloads config
SAVE_DIR = "data"
DATASET_DOWNLOAD_PATH = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
DATASET_FILENAME = "snli"
GLOVE_DOWNLOAD_PATH = "http://nlp.stanford.edu/data/glove.42B.300d.zip"
GLOVE_FILENAME = "glove"


# Checks, whether the unpacked archive is present.
def is_unpacked(archive_name):
    return os.path.isdir(SAVE_DIR + "/" + archive_name)


# Specifically checks for unpacked dataset and glove vectors
def check_all_unpacked():
    return is_unpacked(DATASET_FILENAME) and is_unpacked(GLOVE_FILENAME)


# Returns the path to folder with unpacked dataset.
def unpacked_dataset_path():
    return SAVE_DIR + "/" + DATASET_FILENAME + "/snli_1.0"


# Returns the path to folder with unpacked GloVe word vectors.
def unpacked_glove_path():
    return SAVE_DIR + "/" + GLOVE_FILENAME


# Downloads and saves zip archive.
def download(url, archive_name, label=""):
    if os.path.exists(SAVE_DIR + "/" + archive_name + ".zip") or is_unpacked(archive_name):
        logger.info(label.capitalize() + " already downloaded.")
        return
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    logger.info("Downloading the " + label + " archive from " + url)
    wget.download(url, SAVE_DIR + "/" + archive_name + ".zip")
    logger.success("Download completed.")


# Unzips the compressed dataset.
def unpack(archive_name, label=""):
    if is_unpacked(archive_name):
        logger.info(label.capitalize() + " already unpacked.")
        return
    if not os.path.exists(SAVE_DIR + "/" + archive_name + ".zip"):
        logger.error("No " + label + " zipfile to unpack")
        return

    logger.info("Unpacking " + label)
    os.makedirs(SAVE_DIR + "/" + archive_name)
    with zipfile.ZipFile(SAVE_DIR + "/" + archive_name + ".zip", "r") as file:
        file.extractall(SAVE_DIR + "/" + archive_name)
    logger.success("Unpacking complete.")
    os.remove(SAVE_DIR + "/" + archive_name + ".zip")


def run():
    logger.header("Running downloader module.")
    download(DATASET_DOWNLOAD_PATH, DATASET_FILENAME, "dataset")
    unpack(DATASET_FILENAME, "dataset")
    download(GLOVE_DOWNLOAD_PATH, GLOVE_FILENAME, "glove")
    unpack(GLOVE_FILENAME, "glove")
