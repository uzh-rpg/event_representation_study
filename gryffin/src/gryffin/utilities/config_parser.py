#!/usr/bin/env

__author__ = "Florian Hase"

import numpy as np

from . import GryffinParseError, GryffinValueError
from . import Logger

from . import ParserJSON, CategoryParser
from . import default_general_configuration
from . import default_database_configuration
from . import default_model_configuration
from . import safe_execute


class Configuration(object):
    def __init__(self, me=""):
        self.me = me + ":"
        self.added_props = []
        self.added_attrs = []

    def __str__(self):
        new_line = "%s\n" % self.me
        for prop in sorted(self.added_props):
            new_line += "--> %s:\t%s\n" % (prop, getattr(self, prop))
        return new_line

    def __iter__(self):
        for _ in range(self.num_elements):
            info_dict = {}
            for prop_index, prop in enumerate(self.added_props):
                info_dict[prop] = self.added_attrs[prop_index][_]
            yield info_dict

    def __getitem__(self, index):
        info_dict = {}
        for prop_index, prop in enumerate(self.added_props):
            info_dict[prop] = self.added_attrs[prop_index][index]
        return info_dict

    def to_dict(self):
        return {prop: getattr(self, prop) for prop in sorted(self.added_props)}

    def add_attr(self, prop, attr):
        setattr(self, prop, attr)
        if prop not in self.added_props:
            self.added_props.append(prop)
            self.added_attrs.append(attr)
        try:
            self.num_elements = len(attr)
        except TypeError:
            pass

    def get_attr(self, prop):
        return getattr(self, prop)


class ConfigParser(Logger):
    TYPE_ATTRIBUTES = {
        "continuous": ["low", "high", "periodic"],
        "discrete": ["low", "high", "options", "descriptors"],
        "categorical": ["options", "descriptors", "category_details"],
    }

    def __init__(self, config_file=None, config_dict=None):
        Logger.__init__(self, "ConfigParser", verbosity=3)
        self.category_parser = CategoryParser()
        self.config_file = config_file
        self.config_dict = config_dict

    def _parse_general(self, provided_settings):
        self.general = Configuration("general")
        for general_key, general_value in default_general_configuration.items():
            if general_key in provided_settings:
                general_value = provided_settings[general_key]
            if general_value in ["True", "False"]:
                general_value = general_value == "True"
            self.general.add_attr(general_key, general_value)

    def _parse_database(self, provided_settings):
        self.database = Configuration("database")
        for general_key, general_value in default_database_configuration.items():
            if general_key in provided_settings:
                general_value = provided_settings[general_key]
            if general_value in ["True", "False"]:
                general_value = general_value == "True"
            self.database.add_attr(general_key, general_value)

    def _parse_model(self, provided_settings):
        self.model_details = Configuration("model")
        for general_key, general_value in default_model_configuration.items():
            if general_key in provided_settings:
                general_value = provided_settings[general_key]
            if general_value in ["True", "False"]:
                general_value = general_value == "True"
            self.model_details.add_attr(general_key, general_value)

    def _parse_parameters(self, provided_settings):
        self.parameters = Configuration("parameters")
        self.features = Configuration("features")
        self.kernels = Configuration("kernels")
        param_configs = {
            "name": [],
            "type": [],
            "specifics": [],
            "process_constrained": [],
        }
        feature_configs = {
            "name": [],
            "type": [],
            "specifics": [],
            "process_constrained": [],
        }
        kernel_configs = {
            "name": [],
            "type": [],
            "specifics": [],
            "process_constrained": [],
        }

        if len(provided_settings) == 0:
            self.log("need to define at least one parameter", "FATAL")

        # -----------------------------
        # parse parameter configuration
        # -----------------------------
        for setting in provided_settings:
            num_cats = 1

            # check if constrained
            if "process_constrained" in setting:
                setting["process_constrained"] = bool(setting["process_constrained"])
            else:
                setting["process_constrained"] = False

            if setting["type"] == "continuous":
                # check order
                if setting["high"] <= setting["low"]:
                    GryffinValueError(
                        'upper limit (%f) needs to be larger than the lower limit (%f) for parameter "%s"'
                        % (setting["high"], setting["low"], setting["name"])
                    )

                # check if periodic
                if "periodic" in setting:
                    setting["periodic"] = bool(setting["periodic"])
                else:
                    setting["periodic"] = False

            elif setting["type"] == "discrete":
                # check order
                if setting["high"] <= setting["low"]:
                    GryffinValueError(
                        'upper limit (%f) needs to be larger than the lower limit (%f) for parameter "%s"'
                        % (setting["high"], setting["low"], setting["name"])
                    )
                setting["options"] = np.arange(
                    setting["low"], setting["high"] + 1, dtype=np.int32
                )  # +1 to use closed interval
                setting["descriptors"] = np.arange(
                    0, setting["high"] - setting["low"] + 1, dtype=np.float64
                )  # +1 to use closed interval
                setting["descriptors"] = np.reshape(
                    setting["descriptors"], (len(setting["descriptors"]), 1)
                )
                num_cats = len(setting["options"])

            elif setting["type"] == "categorical":
                if "category_details" not in setting:
                    self.log(
                        "`category_details` needs to be defined for a categorical variable to know "
                        "which options to be considered",
                        "FATAL",
                    )

                options, descriptors = self.category_parser.parse(
                    setting["category_details"]
                )
                setting["options"] = options
                setting["descriptors"] = descriptors

                # if someone is requesting the descriptor transformation but descriptors are not provided,
                #  something is wrong and the user needs to fix the input
                if self.general.auto_desc_gen is True and descriptors is None:
                    self.log(
                        "Automatic descriptor generation requested, but no descriptors provided for "
                        'parameter "%s".' % setting["name"],
                        "FATAL",
                    )

                num_cats = len(setting["options"])
            else:
                self.log(
                    'Did not understand parameter type "%s" for parameter "%s". Please choose from "continuous", "discrete" or "categorical".'
                    % (setting["type"], setting["name"]),
                    "FATAL",
                )

            # ---------------------
            # populate config dicts
            # ---------------------
            for key in param_configs.keys():
                if key == "specifics":
                    element = {
                        spec_key: setting[spec_key]
                        for spec_key in self.TYPE_ATTRIBUTES[setting["type"]]
                    }
                else:
                    element = setting[key]

                param_configs[key].append(element)
                feature_configs[key].append(element)
                kernel_configs[key].extend([element for _ in range(num_cats)])

        # write configuration
        for key in param_configs.keys():
            self.parameters.add_attr(key, param_configs[key])
            self.features.add_attr(key, feature_configs[key])
            self.kernels.add_attr(key, kernel_configs[key])

    def _parse_objectives(self, provided_settings):
        """
        Note that we expect objectives to be provided in order or priority/hierarchy
        """
        self.objectives = Configuration("objectives")
        obj_configs = {"name": [], "goal": [], "tolerance": [], "absolute": []}

        if len(provided_settings) == 0:
            self.log("need to define at least one objective", "FATAL")

        # if single objective
        elif len(provided_settings) == 1:
            setting = provided_settings[0]
            obj_configs["name"].append(setting["name"])
            obj_configs["goal"].append(setting["goal"])
            obj_configs["tolerance"].append(0.0)
            obj_configs["absolute"].append(False)
        # if multiple objective
        else:
            for setting in provided_settings:
                for key in obj_configs.keys():
                    if key not in setting:
                        self.log(
                            f'you need to define "{key}" in multi-objective optimizations',
                            "FATAL",
                        )
                    obj_configs[key].append(setting[key])

        # write configuration
        for key in obj_configs:
            self.objectives.add_attr(key, obj_configs[key])

    @property
    def settings(self):
        settings_dict = {}
        settings_dict["general"] = self.general.to_dict()
        settings_dict["database"] = self.database.to_dict()
        settings_dict["parameters"] = self.parameters.to_dict()
        settings_dict["objectives"] = self.objectives.to_dict()
        return settings_dict

    @property
    def process_constrained(self):
        is_constrained = np.any(self.parameters.process_constrained)
        return is_constrained

    @property
    def param_names(self):
        return self.parameters.name

    @property
    def param_types(self):
        return self.parameters.type

    @property
    def continuous_mask(self):
        return np.array(
            [True if p == "continuous" else False for p in self.param_types]
        )

    @property
    def discrete_mask(self):
        return np.array([True if p == "discrete" else False for p in self.param_types])

    @property
    def categorical_mask(self):
        return np.array(
            [True if p == "categorical" else False for p in self.param_types]
        )

    @property
    def param_options(self):
        options = []
        for spec in self.parameters.specifics:
            if "options" in spec:
                options.append(spec["options"])
            else:
                options.append(None)
        return options

    @property
    def param_periodic(self):
        periodic = []
        for spec in self.parameters.specifics:
            if "periodic" in spec:
                periodic.append(spec["periodic"])
            else:
                periodic.append(False)
        return periodic

    @property
    def param_lowers(self):
        lowers = []
        for spec in self.features.specifics:
            # if low in spec ==> continuous or discrete
            if "low" in spec:
                lowers.append(spec["low"])
            # otherwise it's categorical
            elif "options" in spec:
                lowers.append(0.0)
        return np.array(lowers)

    @property
    def param_uppers(self):
        uppers = []
        for spec in self.features.specifics:
            # if low in spec ==> continuous or discrete
            if "high" in spec:
                uppers.append(spec["high"])
            # otherwise it's categorical
            elif "options" in spec:
                num_options = len(spec["options"])
                uppers.append(num_options - 1.0)
        return np.array(uppers)

    @property
    def feature_descriptors(self):
        descriptors = []
        for spec in self.features.specifics:
            if "descriptors" in spec:
                descriptors.append(spec["descriptors"])
            else:
                descriptors.append(None)
        return np.array(descriptors)

    @property
    def feature_process_constrained(self):
        return self.features.process_constrained

    @property
    def feature_lengths(self):
        lengths = []
        for spec in self.features.specifics:
            if "options" in spec:
                lengths.append(len(spec["options"]))
            else:
                lengths.append(spec["high"] - spec["low"])
        return np.array(lengths)

    @property
    def feature_lowers(self):
        lowers = []
        for spec, ftype in zip(self.features.specifics, self.feature_types):
            if "low" in spec:
                if ftype == "discrete":
                    lowers.append(0.0)  # discrete features are shifted to zero
                else:
                    lowers.append(spec["low"])
            else:
                lowers.append(0)
        return np.array(lowers)

    @property
    def feature_uppers(self):
        uppers = []
        for spec, ftype in zip(self.features.specifics, self.feature_types):
            if "high" in spec:
                if ftype == "discrete":
                    upper = spec["high"] - spec["low"]
                    uppers.append(upper)  # discrete features are shifted to zero
                else:
                    uppers.append(spec["high"])
            else:
                uppers.append(1.0)
        return np.array(uppers)

    @property
    def feature_names(self):
        return self.features.name

    @property
    def feature_options(self):
        options = []
        for spec in self.features.specifics:
            if "options" in spec:
                options.append(spec["options"])
            else:
                options.append(None)
        return options

    @property
    def feature_ranges(self):
        return self.feature_uppers - self.feature_lowers

    @property
    def feature_sizes(self):
        sizes = []
        for feature_index, feature_type in enumerate(self.feature_types):
            if feature_type == "categorical":
                feature_size = len(self.features.specifics[feature_index]["options"])
            elif feature_type == "discrete":
                feature_size = len(self.features.specifics[feature_index]["options"])
            else:
                feature_size = 1
            sizes.append(feature_size)
        return np.array(sizes)

    @property
    def feature_types(self):
        return self.features.type

    @property
    def feature_periodic(self):
        periodic = []
        for spec in self.features.specifics:
            if "periodic" in spec:
                periodic.append(spec["periodic"])
            else:
                periodic.append(False)
        return periodic

    @property
    def num_features(self):
        return len(self.feature_names)

    @property
    def kernel_names(self):
        return self.kernels.name

    @property
    def kernel_sizes(self):
        sizes = []
        for kernel_index, kernel_type in enumerate(self.kernel_types):
            if kernel_type == "categorical":
                kernel_size = len(self.kernels.specifics[kernel_index]["options"])
            elif kernel_type == "discrete":
                kernel_size = len(self.kernels.specifics[kernel_index]["options"])
            elif kernel_type == "continuous":
                kernel_size = 1
            else:
                raise NotImplementedError()
            sizes.append(kernel_size)
        return np.array(sizes)

    @property
    def kernel_types(self):
        return self.kernels.type

    @property
    def kernel_lowers(self):
        lowers = []
        for spec in self.kernels.specifics:
            if "options" in spec:
                lowers.append(0.0)
            else:
                lowers.append(spec["low"])
        return np.array(lowers)

    @property
    def kernel_uppers(self):
        uppers = []
        for spec in self.kernels.specifics:
            if "options" in spec:
                uppers.append(1.0)
            else:
                uppers.append(spec["high"])
        return np.array(uppers)

    @property
    def kernel_ranges(self):
        return self.kernel_uppers - self.kernel_lowers

    @property
    def kernel_periodic(self):
        periodic = []
        for spec in self.kernels.specifics:
            if "periodic" in spec:
                periodic.append(spec["periodic"])
            else:
                periodic.append(False)
        return periodic

    # ------------------------------------
    # Properties related to the objectives
    # ------------------------------------
    @property
    def obj_names(self):
        return self.objectives.name

    @property
    def obj_tolerances(self):
        return self.objectives.tolerance

    @property
    def obj_goals(self):
        return self.objectives.goal

    @property
    def obj_absolutes(self):
        return self.objectives.absolute

    def get(self, attr):
        return self.general.get_attr(attr)

    def get_db(self, attr):
        return self.database.get_attr(attr)

    def set_home(self, home_path):
        self.general.add_attr("home", home_path)

    def _parse(self, config_dict):
        self.config = config_dict

        if "general" in self.config:
            self._parse_general(self.config["general"])
        else:
            self._parse_general({})
        self.update_verbosity(self.general.verbosity)

        if "database" in self.config:
            self._parse_database(self.config["database"])
        else:
            self._parse_database({})

        if "model" in self.config:
            self._parse_model(self.config["model"])
        else:
            self._parse_model({})

        self._parse_parameters(self.config["parameters"])
        self._parse_objectives(self.config["objectives"])

    @safe_execute(GryffinParseError)
    def parse_config_file(self, config_file=None):
        if config_file is not None:
            self.config_file = config_file

        self.json_parser = ParserJSON(json_file=self.config_file)
        self.config_dict = self.json_parser.parse()
        self._parse(self.config_dict)

    @safe_execute(GryffinParseError)
    def parse_config_dict(self, config_dict=None):
        if config_dict is not None:
            self.config_dict = config_dict

        self._parse(self.config_dict)

    def parse(self):
        # test if both dict and file have been provided
        if self.config_dict is not None and self.config_file is not None:
            self.log(
                "Found both configuration file and configuration dictionary. Will parse configuration from dictionary and ignore file",
                "WARNING",
            )
            self.parse_config_dict(self.config_dict)
        elif self.config_dict is not None:
            self.parse_config_dict(self.config_dict)
        elif self.config_file is not None:
            self.parse_config_file(self.config_file)
        else:
            self.log(
                "Cannot parse configuration due to missing configuration file or configuration dictionary",
                "ERROR",
            )
