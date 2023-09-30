#!/usr/bin/env python

import resource
import sys


def parse_time(start, end):
    elapsed = end - start  # elapsed time in seconds
    if elapsed <= 1.0:
        ms = elapsed * 1000.0
        time_string = f"{ms:.1f} ms"
    elif 1.0 < elapsed < 60.0:
        time_string = f"{elapsed:.1f} s"
    else:
        m, s = divmod(elapsed, 60)
        time_string = f"{m:.0f} min {s:.0f} s"
    return time_string


def memory_usage():
    if sys.platform == "darwin":  # MacOS --> memory in bytes
        kB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1000.0
    else:  # Linux --> memory in kilobytes
        kB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    GB = kB // 1000000.0
    MB = kB // 1000.0
    kB = kB % 1000.0
    return GB, MB, kB


def print_memory_usage():
    GB, MB, kB = memory_usage()
    print(f"==> MEMORY USAGE: {GB:.0f} GB, {MB:.0f} MB, {kB:.0f} kB")
