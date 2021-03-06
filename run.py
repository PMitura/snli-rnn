"""
Runner program for all routines. Possible parameters (more than one applicable):
    download:   If not present, downloads and unpacks the SNLI dataset from SNLP website.
    preprocess: Precompute word embeddings and prepare dataset
    train:      Runs training phase of the neural network.
    test:       Runs test data on pretrained model (if present).
"""

import sys
import rnn


def main(argv):

    if not len(argv):
        rnn.downloader.run()
        rnn.preprocessor.run()
        rnn.trainer.run()
        return

    for arg in argv:
        if arg == "download":
            rnn.downloader.run()
        elif arg == "preprocess":
            rnn.preprocessor.run()
        elif arg == "train":
            rnn.trainer.run()
        elif arg == "test":
            print("Test: not implemented yet.")
        else:
            print(arg + ": unrecognized command.")


if __name__ == '__main__':
    main(sys.argv[1:])
