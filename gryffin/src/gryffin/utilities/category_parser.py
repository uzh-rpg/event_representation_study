#!/usr/bin/env

__author__ = "Florian Hase, Matteo Aldeghi"

import numpy as np
import pickle
import json


class CategoryParser:
    def __init__(self):
        pass

    def parse(self, file_name):
        if isinstance(file_name, dict):
            options, descriptors = self.parse_dict(file_name)
        elif isinstance(file_name, str):
            options, descriptors = self.parse_file(file_name)
        else:
            NotImplementedError(f"cannot parse object of type {type(file_name)}")

        if descriptors is not None:
            # -----------------------
            # rescale the descriptors
            # -----------------------
            min_descriptors, max_descriptors = np.amin(descriptors, axis=0), np.amax(
                descriptors, axis=0
            )
            # explicitly avoid division by zero
            # if we have descriptor ranges that are zero, the user needs to revise the problem design
            range_min_descriptors = max_descriptors - min_descriptors
            for i, r in enumerate(range_min_descriptors):
                if r < 10e-6:
                    ValueError(f"the range of the {i}th descriptor provided is zero!")
            descriptors = (descriptors - min_descriptors) / (
                max_descriptors - min_descriptors
            )

        return options, descriptors

    @staticmethod
    def parse_dict(cat_details):
        """Dict parser for options and descriptors. The dictionary is expected to contain options as keys
        and descriptors as values, e.g. ``{'A':[1, 2], 'B':[5,6]}``. If there are no descriptors
        to be provided, use empty lists or None as values, e.g. ``{'A':None, 'B':None}`` or ``{'A':[], 'B':[]}``.
        """
        options = list(cat_details.keys())
        descriptors = list(cat_details.values())
        if len(descriptors) == 0 or None in descriptors:
            descriptors = None
        else:
            descriptors = np.array(descriptors)
        return options, descriptors

    def parse_file(self, file_name):
        suffix = file_name.split(".")[-1]
        if suffix == "json":
            options, descriptors = self._parse_json(file_name)
        elif suffix == "pkl":
            options, descriptors = self._parse_pickle(file_name)
        elif suffix in ["xlsx", "xls"]:
            options, descriptors = self._parse_excel(file_name)
        elif suffix == "csv":
            options, descriptors = self._parse_csv(file_name)
        else:
            NotImplementedError(f"cannot parse files with extension {suffix}")

        return options, descriptors

    def _parse_pickle(self, file_name):
        """Pickle parser for options and descriptors. The pickle file is expected to contain a dictionary where the keys
        are the options and the values are the descriptors, e.g. ``{'A':[1, 2], 'B':[5,6]}``. If there are no descriptors
        to be provided, use empty lists or None as values, e.g. ``{'A':None, 'B':None}`` or ``{'A':[], 'B':[]}``.
        """
        with open(file_name, "rb") as content:
            cat_details = pickle.load(content)

        options, descriptors = self.parse_dict(cat_details)
        return options, descriptors

    def _parse_json(self, file_name):
        """JSON parser for options and descriptors. The JSON file is expected to contain a dictionary where the keys
        are the options and the values are the descriptors, e.g. ``{'A':[1, 2], 'B':[5,6]}``. If there are no descriptors
        to be provided, use empty lists or None as values, e.g. ``{'A':None, 'B':None}`` or ``{'A':[], 'B':[]}``.
        """
        with open(file_name) as content:
            cat_details = json.loads(content.read())

        options, descriptors = self.parse_dict(cat_details)
        return options, descriptors

    @staticmethod
    def _parse_csv(file_name):
        """CSV parser for options and descriptors. The expected format contains each option and associated descriptors
        in each row. The first column contains the name of the option, and all other columns the descriptors.
        """
        import pandas as pd  # import pandas only if needed

        df = pd.read_csv(file_name, header=None)
        options = df.iloc[:, 0].to_list()
        descriptors = df.iloc[:, 1:].to_numpy()
        if descriptors.size == 0:
            descriptors = None
        return options, descriptors

    @staticmethod
    def _parse_excel(file_name):
        """Excel parser for options and descriptors. The expected format contains each option and associated descriptors
        in each row. The first column contains the name of the option, and all other columns the descriptors.
        """
        import pandas as pd  # import pandas only if needed

        df = pd.read_excel(file_name, header=None)
        options = df.iloc[:, 0].to_list()
        descriptors = df.iloc[:, 1:].to_numpy()
        if descriptors.size == 0:
            descriptors = None
        return options, descriptors
