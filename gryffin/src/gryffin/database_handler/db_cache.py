#!/usr/bin/env python

__author__ = "Florian Hase"

# ========================================================================

from gryffin.utilities import Logger

# ========================================================================


class DB_Cache(Logger):
    def __init__(self, attributes, entries=[], verbosity=0):
        Logger.__init__(self, "DB_Cache", verbosity=verbosity)
        self.attributes = attributes

        self.cache = {attr: [] for attr in self.attributes}
        self.num_items = 0
        for entry in entries:
            self.add(entry)

    def __getitem__(self, item):
        try:
            return self.cache[item]
        except KeyError:
            return []

    def add(self, info_dict):
        for attr in self.attributes:
            if attr in info_dict:
                self.cache[attr].append(info_dict[attr])
            else:
                self.cache[attr].append(None)
        self.num_items += 1

    def fetch_all(self, condition_dict):
        results = []
        for element_index in range(self.num_items):
            for key, value in condition_dict.items():
                if value != self.cache[key][element_index]:
                    break
            else:
                result = {
                    attr: self.cache[attr][element_index] for attr in self.attributes
                }
                results.append(result)
        return results

    def update_all(self, condition_dict, update_dict):
        for element_index in range(self.num_items):
            for key, value in condition_dict.items():
                if value != self.cache[key][element_index]:
                    break
            else:
                for key, value in update_dict.items():
                    self.cache[key][element_index] = value
