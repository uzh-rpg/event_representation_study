#!/usr/bin/env python

__author__ = "Florian Hase, Matteo Aldeghi"

import sys
import traceback
from rich.console import Console


class Logger:
    # DEBUG, INFO           --> stdout
    # WARNING, ERROR, FATAL --> stderr

    VERBOSITY_LEVELS = {
        0: ["FATAL"],
        1: ["FATAL", "ERROR"],
        2: ["FATAL", "ERROR", "WARNING"],
        3: ["FATAL", "ERROR", "WARNING", "STATS"],  # minimal info
        4: ["FATAL", "ERROR", "WARNING", "STATS", "INFO"],  # richer info
        5: ["FATAL", "ERROR", "WARNING", "STATS", "INFO", "DEBUG"],
    }

    WRITER = {
        "DEBUG": sys.stdout,
        "INFO": sys.stdout,
        "WARNING": sys.stderr,
        "ERROR": sys.stderr,
        "FATAL": sys.stderr,
    }

    # more colors and styles:
    # https://stackoverflow.com/questions/2048509/how-to-echo-with-different-colors-in-the-windows-command-line
    # https://joshtronic.com/2013/09/02/how-to-use-colors-in-command-line-output/

    NONE = ""
    WHITE = "#ffffff"
    GREEN = "#d9ed92"
    GRAY = "#d3d3d3"
    YELLOW = "#f9dc5c"
    ORANGE = "#f4a261"
    RED = "#e5383b"
    PURPLE = "#9d4edd"

    COLORS = {
        "DEBUG": GRAY,
        "INFO": NONE,
        "STATS": NONE,
        "WARNING": ORANGE,
        "ERROR": RED,
        "FATAL": PURPLE,
    }

    def __init__(self, name, verbosity=4):
        """
        name : str
            name to give this logger.
        verbosity : int
            verbosity level, between ``0`` and ``4``. with ``0`` only ``FATAL`` messages are shown, with ``1`` also
            ``ERROR``, with ``2`` also ``WARNING``, with ``3`` also ``INFO``, with ``4`` also ``DEBUG``. Default
            is ``3``.
        """
        self.name = name
        self.verbosity = verbosity
        self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]
        self.console = Console(stderr=False)
        self.error_console = Console(stderr=True)

    def update_verbosity(self, verbosity=3):
        self.verbosity = verbosity
        self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]

    def log(self, message, message_type):
        # check if we need to log the message
        if message_type in self.verbosity_levels:
            color = self.COLORS[message_type]
            error_message = None
            if message_type in ["WARNING", "ERROR", "FATAL"]:
                error_message = traceback.format_exc()
                if "NoneType: None" not in error_message:
                    self.error_console.print(error_message, style=f"{color}")

            self.console.print(message, style=f"{color}")
            return error_message, message

    def log_chapter(self, title, line="─", style="#34a0a4"):
        if self.verbosity >= 4:
            title = " " + title + " "
            self.console.print(f"{title:{line}^80}", style=style)
