# %%
import sys
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from classification_ehr_shot import grid_search

from const import root_dir
import argparse
import json


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classification Dispatcher")
    parser.add_argument(
        "--dataset",
        choices=["mimic", "ki_thrust"],
        default="mimic",
        help="dataset to use for evaluation",
    )
    parser.add_argument(
        "--task",
        choices=["mortality", "rehospitalization", "future_disease"],
        default="mortality",
        help="task to evaluate on evaluation",
    )
    parser.add_argument(
        "--encoding",
        type=int,
        default=0,
        help="encoding number to evaluate",
    )
    parser.add_argument(
        "--resample_seed",
        type=int,
        default=0,
        help="bootstrapping resample seed",
    )

    parser.add_argument(
        "--verbose", type=bool, default=False, help="status report print statements"
    )

    args = parser.parse_args()

    data_dir = root_dir.joinpath("data", "datasets")
    save_dir = root_dir.joinpath("data", "results", "classification_stats")

    encoding_paths_mimic = [
        path
        for taskpath in data_dir.iterdir()
        if taskpath.joinpath("encoded").exists()
        for path in taskpath.joinpath("encoded").iterdir()
        if "ki_thrust" != taskpath.name
    ]

    encoding_paths_mimic = list(
        filter(
            lambda path: "mimic_iv_2.2" in str(path) and "_full" in str(path),
            encoding_paths_mimic,
        )
    )

    encoding_paths_ki_thrust = [
        path
        for taskpath in data_dir.joinpath("ki_thrust").iterdir()
        for path in taskpath.iterdir()
    ]

    if args.dataset == "mimic":
        encoding_paths = encoding_paths_mimic
    elif args.dataset == "ki_thrust":
        encoding_paths = encoding_paths_ki_thrust
        if args.task == "mortality" or args.task == "rehospitalization":
            encoding_paths = [
                path for path in encoding_paths if "mort_rehosp" in str(path)
            ]
        else:
            encoding_paths = [
                path for path in encoding_paths if "mort_rehosp" not in str(path)
            ]
    if args.verbose:
        print(f"{len(encoding_paths)} ENCODINGS:", *encoding_paths, sep="\n")

    task_cols = [args.task]
    task_type = "boolean"
    if args.task == "future_disease":
        task_cols = list("ABCDEFGHIJKLMNOPQRSTUVXYZ")
        task_type = "multilabel"
    if args.verbose:
        print("Y_COLS\n", task_cols)

    encoding_path = encoding_paths[args.encoding]
    resample_seed = args.resample_seed
    if args.verbose:
        print("CHOOSEN ENCODING\n", encoding_path)
        print("RESAMPLE SEED\n", resample_seed)

    encoding = encoding_path.name
    dataset_name = encoding_path.parent.parent.name

    encoding_save_dir = save_dir.joinpath(
        args.dataset, args.task, dataset_name, encoding
    )

    data_train_default = pd.read_csv(encoding_path.joinpath("train.csv"))
    data_val = pd.read_csv(encoding_path.joinpath("val.csv"))
    data_test = pd.read_csv(encoding_path.joinpath("test.csv"))

    if args.dataset == "ki_thrust" and args.task == "future_disease":
        data_train_default = data_train_default.rename(columns={"X_W": "W"})
        data_val = data_test.rename(columns={"X_W": "W"})
        data_test = data_test.rename(columns={"X_W": "W"})

    X_cols = [col for col in data_train_default.columns if "X_" in col]

    Y_cols = task_cols

    Y_cols = [col for col in Y_cols if col in data_train_default.columns]

    if args.verbose:
        print(X_cols, Y_cols)

    results = None

    classifiers = {
        "LogisticRegression": {
            "model": LogisticRegression,
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
                "class_weight": ["balanced"],
                "max_iter": [200],
                "random_state": [resample_seed],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier,
            "param_grid": {
                "n_estimators": [10, 25, 50, 100, 250],
                "max_depth": [2, 5, 10, None],
                "class_weight": ["balanced"],
                "random_state": [resample_seed],
            },
        },
    }

    data_train = resample(
        data_train_default,
        replace=True,
        n_samples=len(data_train_default),
        random_state=resample_seed,
    )

    X_train = data_train[X_cols]
    Y_train = data_train[Y_cols]

    X_val = data_val[X_cols]
    Y_val = data_val[Y_cols]

    X_test = data_test[X_cols]
    Y_test = data_test[Y_cols]

    for classifier_name, classifier in classifiers.items():

        print(classifier_name)

        res, model = grid_search(
            classifier["model"],
            X_train,
            Y_train,
            X_val,
            Y_val,
            X_test,
            Y_test,
            classifier["param_grid"],
            encoding,
            task_type=task_type,
        )

        res["resample_seed"] = resample_seed

        results = pd.concat([results, res])

        classifier_save_dir = encoding_save_dir.joinpath(
            "classifiers",
            classifier_name,
            f"resample_seed_{resample_seed}",
        )
        classifier_save_dir.mkdir(parents=True, exist_ok=True)
        if args.dataset != "ki_thrust" or classifier_name != "RandomForest":
            with open(classifier_save_dir.joinpath(f"best_model.pkl"), "wb") as file:
                pickle.dump(model["model"], file)

        with open(classifier_save_dir.joinpath(f"params.json"), "w") as file:
            json.dump(model["params"], file, sort_keys=True, indent=4)

    encoding_save_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        encoding_save_dir.joinpath(f"stats_resample_seed_{resample_seed}.csv")
    )
# %%
