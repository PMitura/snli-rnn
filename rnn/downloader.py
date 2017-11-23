"""
Downloader module, gets and unpacks the dataset if not present.
"""

import os
import wget
import zipfile
import rnn.logger as logger

DOWNLOAD_PATH = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
SAVE_DIR = "data"
DATASET_FILENAME = "snli"


def dataset_unpacked():
    return os.path.isdir(SAVE_DIR + "/" + DATASET_FILENAME)


def download():
    if os.path.exists(SAVE_DIR + "/" + DATASET_FILENAME + ".zip") or dataset_unpacked():
        logger.info("Dataset already downloaded.")
        return
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    logger.info("Downloading the dataset from " + DOWNLOAD_PATH)
    wget.download(DOWNLOAD_PATH, SAVE_DIR + "/" + DATASET_FILENAME + ".zip")
    logger.success("Download completed." + DOWNLOAD_PATH)


def unpack():
    if dataset_unpacked():
        logger.info("Dataset already unpacked.")
        return
    if not os.path.exists(SAVE_DIR + "/" + DATASET_FILENAME + ".zip"):
        logger.error("No dataset zipfile to unpack")
        return

    logger.info("Unpacking dataset.")
    os.makedirs(SAVE_DIR + "/" + DATASET_FILENAME)
    with zipfile.ZipFile(SAVE_DIR + "/" + DATASET_FILENAME + ".zip", "r") as file:
        file.extractall(SAVE_DIR + "/" + DATASET_FILENAME)
    logger.success("Dataset unpacked.")
    os.remove(SAVE_DIR + "/" + DATASET_FILENAME + ".zip")


def run():
    logger.header("Running downloader module.")
    download()
    unpack()
