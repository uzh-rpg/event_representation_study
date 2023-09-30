#!/usr/bin/env python

__author__ = "Florian Hase"

# =======================================================================

import pickle

from utilities import Logger

# =======================================================================


class DB_Writer(Logger):
    def __init__(self, config):
        self.config = config
        Logger.__init__(self, "DB_Writer", self.config.get("verbosity"))

    def write(self, db_content, outfile, out_format):
        pickle.dump(db_content, open(outfile, "wb"))
