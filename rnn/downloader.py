# Downloader module, gets and unpacks the dataset if not present

import os

DOWNLOAD_PATH = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
SAVE_DIR = "data"


def download():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)



def run():
    print("\nRunning downloader module.")
