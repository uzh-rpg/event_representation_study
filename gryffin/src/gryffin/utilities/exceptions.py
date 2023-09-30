#!/usr/bin/env python

__author__ = "Florian Hase"

import sys
import traceback


class AbstractError(Exception):
    def __init__(self, message):
        self.__call__(message)

    def __call__(self, message):
        error_traceback = traceback.format_exc()
        error_traceback = "\n".join(error_traceback.split("\n")[:-2]) + "\n\n"
        error_type = "\x1b[0;31m%s: %s\x1b[0m\n" % (self.name, message)

        if "SystemExit" in error_traceback:
            return None

        sys.stderr.write(error_traceback)
        sys.stderr.write(error_type)
        sys.exit()


class GryffinModuleError(AbstractError):
    name = "GryffinModuleError"


class GryffinNotFoundError(AbstractError):
    name = "GryffinNotFoundError"


class GryffinParseError(AbstractError):
    name = "GryffinParseError"


class GryffinSettingsError(AbstractError):
    name = "GryffinSettingsError"


class GryffinUnknownSettingsError(AbstractError):
    name = "GryffinUnknownSettingsError"


class GryffinValueError(AbstractError):
    name = "GryffinValueError"


class GryffinVersionError(AbstractError):
    name = "GryffinVersionError"


class GryffinComputeError(AbstractError):
    name = "GryffinComputeError"
