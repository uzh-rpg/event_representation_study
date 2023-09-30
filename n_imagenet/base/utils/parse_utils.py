from collections import namedtuple
import configparser
from ast import literal_eval
from pathlib import Path

NO_PARSE = ["name", "load_model"]  # List of names to avoid parsing
EXCEPT_NONE = ["load_model"]


def parse_ini(config_path: str):
    read_config = configparser.ConfigParser()
    config_name = Path(config_path).stem

    read_config.read(config_path)
    config_attribs = []
    data_dict = {}

    for section in read_config.sections():
        for key, value in read_config.items(section):
            config_attribs.append(key)
            data_dict[key] = parse_value(value) if key not in NO_PARSE else value
            if key in EXCEPT_NONE and value == "None":  # Account for None
                data_dict[key] = None

        # Modify name just in case of errors
        data_dict["name"] = config_name

    Config = namedtuple("Config", config_attribs)
    cfg = Config(**data_dict)
    return cfg


def parse_value(value):
    if (
        value.replace(".", "", 1)
        .replace("+", "", 1)
        .replace("-", "", 1)
        .replace("e", "", 1)
        .isdigit()
    ):
        # Exponential format and decimal format should be accounted for
        return literal_eval(value)
    elif value == "True" or value == "False":
        if value == "True":
            return True
        else:
            return False
    elif value == "None":
        return None
    elif "," in value:  # Config contains lists
        is_number = any(char.isdigit() for char in value.split(",")[0])
        items_list = value.split(",")

        if "" in items_list:
            items_list.remove("")
        if is_number:
            return [literal_eval(val) for val in items_list]
        else:
            if '"' in items_list[0] and "'" in items_list[0]:
                return [literal_eval(val.strip()) for val in items_list]
            else:
                return [val.strip() for val in items_list]
    else:
        return value
