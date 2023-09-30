#!/usr/bin/env python

__author__ = "Florian Hase"

# =======================================================================

import numpy as np
import pandas as pd
from datetime import datetime

from utilities import Logger

# =======================================================================


class Writer(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def save(self):
        pass


class CsvWriter(Writer):
    def __init__(self, file_name):
        Writer.__init__(self, file_name)

    def __call__(self, pd_frame):
        pd_frame.to_csv(self.file_name)


class ExcelWriter(Writer):
    def __init__(self, file_name):
        Writer.__init__(self, file_name)
        self.writer = pd.ExcelWriter(self.file_name)

    def __call__(self, pd_frame):
        pd_frame.to_excel(self.writer, "Sheet1")

    def save(self):
        self.writer.save()


# =======================================================================


class DB_Writer(Logger):
    def __init__(self, config):
        self.config = config
        Logger.__init__(self, "DB_Writer", self.config.get("verbosity"))

    def create_writer(self, file_name, out_format):
        if out_format == "xlsx":
            self.writer = ExcelWriter(file_name)
        elif out_format == "csv":
            self.writer = CsvWriter(file_name)

    def write(self, db_content, outfile, out_format):
        # create the writer
        self.create_writer(outfile, out_format)

        # convert output dict and save via pandas routine
        dataframe = pd.DataFrame.from_dict(db_content)
        self.writer(dataframe)
        self.writer.save()
