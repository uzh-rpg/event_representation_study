#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import shutil
import json
import argparse
from rich.console import Console
from rich.table import Table
from gryffin import Gryffin
from gryffin.utilities import GryffinSettingsError


# =============
# Parse Options
# =============
def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        dest="file",
        type=str,
        help="Excel/CSV file with all previous experiments.",
        required=True,
    )
    parser.add_argument(
        "-c",
        dest="json",
        type=str,
        help="Json configuration file with parameters and objectives.",
        required=True,
    )
    parser.add_argument(
        "-n",
        metavar="",
        dest="num_experiments",
        type=int,
        help="Number of experiments to suggest. Default is 1. "
        "Note that Gryffin will alternate between exploration and exploitation.",
        default=1,
    )
    parser.add_argument(
        "--num_cpus",
        metavar="",
        dest="num_cpus",
        type=int,
        help="Number of CPUs to use. Default is 1.",
        default=1,
    )
    parser.add_argument(
        "--optimizer",
        metavar="",
        dest="optimizer",
        type=str.lower,
        help='Algorithm to use to optimize the acquisition function. Choices are "adam" or "genetic". '
        'Default is "adam".',
        default="adam",
        choices=["adam", "genetic"],
    )
    parser.add_argument(
        "--dynamic",
        dest="dynamic",
        help="Whether to use dynamic Gryffin. Default is False.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--feas_approach",
        metavar="",
        dest="feas_approach",
        type=str.lower,
        help="Approach to unknown feasibility constraints. Choices are: "
        '"fwa" (feasibility-weighted acquisition), '
        '"fca" (feasibility-constrained acquisition), '
        '"fia" (feasibility-interpolated acquisition). '
        'Default is "fia".',
        default="fia",
        choices=["fwa", "fca", "fia"],
    )
    parser.add_argument(
        "--boosted",
        dest="boosted",
        help="Whether to use boosting. Default is False.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--cached",
        dest="cached",
        help="Whether to use caching. Default is False.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        metavar="",
        dest="random_seed",
        type=int,
        help="Random seed used for initialization. Default is 42.",
        default=42,
    )
    args = parser.parse_args()
    return args


# ====
# Main
# ====
def main(args):
    # load past experiments
    infile_extension = args.file.split(".")[-1]  # get extension
    df_in = _load_tabular_data(args, infile_extension)

    # load params/objectives
    with open(args.json, "r") as jsonfile:
        config = json.load(jsonfile)

    # check we have all right params/objs in the csv file
    obj_names = [obj["name"] for obj in config["objectives"]]  # N.B. order matters
    param_names = [
        param["name"] for param in config["parameters"]
    ]  # N.B. order matters
    _check_table_against_config(args, df_in, obj_names, param_names)

    # drop rows with NaN values in the parameters
    df_in = df_in.dropna(subset=param_names)

    # show past experiments
    print()
    print_df_as_rich_table(df_in, title="Past Experiments")

    # init gryffin
    gryffin = init_objects(args, config)

    # ---------------------
    # Run Gryffin recommend
    # ---------------------
    if len(df_in) == 0:
        observations = []
        samples = suggest_next_experiments(gryffin, observations, args.num_experiments)
    else:
        # build observation list for Gryffin
        observations = _df_to_observations(df_in)

        # ask for next experiments
        samples = suggest_next_experiments(gryffin, observations, args.num_experiments)

    # ---------------------
    # Save samples to file
    # ---------------------

    # create df_samples
    df_samples = pd.DataFrame(columns=df_in.columns)

    for param_name in param_names:
        param_values = [sample[param_name] for sample in samples]
        df_samples.loc[:, param_name] = param_values

    # show proposed experiments
    print_df_as_rich_table(df_samples, title="Proposed Experiments")
    print()

    # append df_samples to df_in
    df_out = df_in.append(df_samples, ignore_index=True, sort=False)

    # make backup of result file
    bkp_file = f"backup_{args.file}"
    if os.path.isfile(bkp_file):
        os.remove(bkp_file)
    shutil.copy(args.file, bkp_file)

    # save new result file
    if infile_extension == "csv":
        df_out.to_csv(args.file, index=False)
    elif infile_extension in ["xls", "xlsx"]:
        df_out.to_excel(args.file, index=False)

    # print final remakrs
    console = Console()
    console.print("Notes:", style="bold red")
    console.print(
        f"- The original file [green]{args.file}[/green] has been backed up as [green]backup_{args.file}[/green]"
    )
    console.print(
        f"- The proposed experiments have been appended to the file [green]{args.file}[/green]"
    )
    obj_names_string = ", ".join(obj_names)
    console.print(
        f"- Add the results for the proposed experiments under the columns [green]{obj_names_string}[/green]"
    )
    print()


# =========
# Functions
# =========
def _load_tabular_data(args, infile_extension):
    # load data
    if infile_extension == "csv":
        df_in = pd.read_csv(args.file)
    elif infile_extension in ["xls", "xlsx"]:
        df_in = pd.read_excel(args.file)

    # rm rows if NaN in parameters. NaN in objective is allowed as infeasible experiment.
    return df_in


def _check_table_against_config(args, df_in, obj_names, param_names):
    """Check inputs for correctness"""
    for obj_name in obj_names:
        if obj_name not in df_in.columns:
            raise ValueError(
                f"Expected objective '{obj_name}' missing from {args.file}"
            )
    for param_name in param_names:
        if param_name not in df_in.columns:
            raise ValueError(
                f"Expected parameter '{param_name}' missing from {args.file}"
            )


def _df_to_observations(df):
    observations = []
    for index, row in df.iterrows():
        d = {}
        for col in df.columns:
            d[col] = row[col]
        observations.append(d)
    return observations


def infer_batches_and_strategies(num_experiments):
    # if num_experiments <= 2, we use 2 strategies either in parallel or sequence (done in suggest_next_experiments)
    if num_experiments <= 2:
        batches = 1
        sampling_strategies = 2
    # if 2 < num_experiments <= 5, use as many sampling strategies as num experiments in one batch
    elif 2 < num_experiments <= 5:
        batches = 1
        sampling_strategies = num_experiments
    # if num_experiments is > 5, figure out a best split
    elif 2 < num_experiments <= 21:
        # if multiple of 5
        if num_experiments % 5 == 0:
            sampling_strategies = 5
            batches = num_experiments / 5
            return batches, sampling_strategies
        # if multiple of 4
        if num_experiments % 4 == 0:
            sampling_strategies = 4
            batches = num_experiments / 4
            return batches, sampling_strategies
        # if multiple of 3
        if num_experiments % 3 == 0:
            sampling_strategies = 3
            batches = num_experiments / 3
            return batches, sampling_strategies
        # if multiple of 2
        if num_experiments % 2 == 0:
            sampling_strategies = 2
            batches = num_experiments / 2
            return batches, sampling_strategies
        raise GryffinSettingsError("please do not select a prime number of experiments")
    else:
        raise GryffinSettingsError(
            "you are selecting a large batch of experiments - this is not the intended use of "
            "Gryffin. Contanct the authors for further guidance if needed."
        )

    return batches, sampling_strategies


def init_objects(args, config):
    batches, sampling_strategies = infer_batches_and_strategies(args.num_experiments)

    if args.feas_approach in ["fia", "fwa"]:
        feas_param = 1
    elif args.feas_approach == "fca":
        feas_param = 0.5

    config["general"] = {
        "num_cpus": args.num_cpus,
        "boosted": args.boosted,
        "caching": args.cached,
        "batches": int(batches),
        "sampling_strategies": int(sampling_strategies),
        "auto_desc_gen": args.dynamic,
        "feas_approach": args.feas_approach,
        "feas_param": feas_param,
        "random_seed": args.random_seed,
        "save_database": False,
        "acquisition_optimizer": args.optimizer,
        "verbosity": 3,
    }

    gryffin = Gryffin(config_dict=config)
    return gryffin


def suggest_next_experiments(gryffin, observations, num_experiments):
    # i.e. sequential experiments with sampling_strategies==2, batches==1
    if num_experiments == 1:
        # select alternating sampling strategy
        sampling_strategies = [-1, 1]
        select_idx = len(observations) % 2
        strategy = sampling_strategies[select_idx]
        samples = gryffin.recommend(
            observations=observations, sampling_strategies=[strategy]
        )
    # i.e. sampling_strategies * batches == num_experiments
    else:
        samples = gryffin.recommend(observations=observations)

    return samples


def print_df_as_rich_table(df, title):
    console = Console()

    table = Table(show_header=True, header_style="bold red", title=title)
    table.add_column("N")
    for col in df.columns:
        table.add_column(col)

    np_data = df.to_numpy()
    for i, row in enumerate(np_data):
        row_str = [f"{i + 1:d}"] + [f"{x:f}" for x in row]
        table.add_row(*row_str)

    console.print(table)


def entry_point():
    args = parse_options()
    main(args)


if __name__ == "__main__":
    entry_point()
