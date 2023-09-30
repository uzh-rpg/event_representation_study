#!/usr/bin/env

__author__ = "Florian Hase"

# ========================================================================

import pickle

# ========================================================================


class ParserPickle(object):
    def __init__(self, pickle_file=None):
        self.pickle_file = pickle_file

    def parse(self, pickle_file=None):
        # update pickle file
        if not pickle_file is None:
            self.pickle_file = pickle_file

        # parse configuration
        if not self.pickle_file is None:
            with open(self.pickle_file, "rb") as content:
                self.parsed_pickle = pickle.load(content)
        return self.parsed_pickle
