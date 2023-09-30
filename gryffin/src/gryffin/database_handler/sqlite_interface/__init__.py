#!/usr/bin/env python

__author__ = "Florian Hase"

# =======================================================================

import sys

from utilities import GryffinModuleError

# =======================================================================

try:
    import sqlalchemy as sql
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = "\n\tTry installing the sqlalchemy package or use a different database framework instead."
    GryffinModuleError(str(error_message) + extension)

# =======================================================================

from .sqlite_operations import AddEntry, FetchEntries, UpdateEntries
from .sqlite_database import SqliteDatabase
