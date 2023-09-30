#!/usr/bin/env python

__author__ = "Florian Hase"

# =======================================================================

import sys

from utilities import PhoenicsModuleError

# =======================================================================

try:
    import pandas
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = "\n\tTry installing the pandas package of use a different database output format (e.g. pickle)."
    PhoenicsModuleError(str(error_message) + extension)

try:
    import openpyxl
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = "\n\tWriting to xlsx requires the openpyxl package. Please install the package or choose a different output format (e.g. csv or pickle)"
    PhoenicsModuleError(str(error_message) + extension)

# =======================================================================

from DatabaseHandler.PandasWriters.db_writer import DB_Writer
