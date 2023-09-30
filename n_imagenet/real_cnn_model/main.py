import argparse
import subprocess
import configparser
from collections import namedtuple
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "n_imagenet"))

from base.utils.parse_utils import parse_ini, parse_value
from real_cnn_model.data.data_container import ImageNetContainer
from real_cnn_model.models.model_container import CNNContainer
from real_cnn_model.train.trainer import CNNTrainer

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    parser = argparse.ArgumentParser(description="Pretrain")
    parser.add_argument(
        "--config", default="./config.ini", help="Config .ini file directory"
    )
    parser.add_argument("--clean", action="store_true", help="Clean experiments")
    parser.add_argument(
        "--background", action="store_true", help="Run experiment in the background"
    )
    parser.add_argument(
        "--override", default=None, help="Arguments for overriding config"
    )

    args = parser.parse_args()

    cfg = parse_ini(args.config)

    # Display config
    print("Display Config:")
    print(open(args.config, "r").read())

    # Confirm config to prevent errors
    if not args.background:
        choice = "y"
        if choice != "y":
            print("Exit!")
            exit()

    if args.override is not None:
        equality_split = args.override.split("=")
        num_equality = len(equality_split)
        assert num_equality > 0
        if num_equality == 2:
            override_dict = {equality_split[0]: parse_value(equality_split[1])}
        else:
            keys = [equality_split[0]]  # First key
            keys += [
                equality.split(",")[-1] for equality in equality_split[1:-1]
            ]  # Other keys
            values = [
                equality.replace("," + key, "")
                for equality, key in zip(equality_split[1:-1], keys[1:])
            ]  # Get values other than last field
            values.append(equality_split[-1])  # Get last value
            values = [value.replace("[", "").replace("]", "") for value in values]

            override_dict = {
                key: parse_value(value) for key, value in zip(keys, values)
            }

        cfg_dict = cfg._asdict()

        Config = namedtuple(
            "Config", tuple(set(cfg._fields + tuple(override_dict.keys())))
        )

        cfg_dict.update(override_dict)

        cfg = Config(**cfg_dict)
        cfg = cfg._replace(name=cfg.name + "_" + args.override)

    # Make instance of data container
    data_container = ImageNetContainer(cfg)

    # Make instance of model container
    model_container = CNNContainer(cfg)

    # Make instance of trainer
    trainer = CNNTrainer(cfg, model_container, data_container)

    config = configparser.ConfigParser()
    config.add_section("Default")

    cfg_dict = cfg._asdict()

    for key in cfg_dict:
        if key != "name":
            config["Default"][key] = (
                str(cfg_dict[key]).replace("[", "").replace("]", "")
            )
        else:
            config["Default"][key] = str(cfg_dict[key])

    with open(trainer.exp_save_dir / "config.ini", "w") as configfile:
        config.write(configfile)

    # Display model
    print(model_container.models["model"])

    if args.clean:
        print("Cleaning experiments!")
        subprocess.call(["rm", "-rf", trainer.exp_save_dir])
        exit()

    trainer.run()


if __name__ == "__main__":
    main()
