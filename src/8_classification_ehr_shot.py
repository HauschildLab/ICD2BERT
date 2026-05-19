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

# from sklearn.dummy import DummyClassifier
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
import argparse

from const import root_dir

from typing import Dict

# from encoder import BertEncoder, CategoricalEncoder, Pat2VecEncoder, BehrtEncoder
import json
from enum import Enum

# import torch_directml, torch
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
capture = None


class task_types(Enum):
    BINARY = "boolean"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


def apply_to_cols(funk, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
    cols = range(y_pred.shape[1])
    return [funk(y_true[:, i], y_pred[:, i], **kwargs) for i in cols]


def prepare_proba_for_auc(proba, trained_classes, all_classes, y_true):
    """
    Returns a probability matrix aligned with y_true for roc_auc_score.
    """
    n_samples = proba.shape[0]

    # Step 1: expand missing training classes → full class set
    full = np.zeros((n_samples, len(all_classes)))
    for i, cls in enumerate(trained_classes):
        idx = all_classes.index(cls)
        full[:, idx] = proba[:, i]

    # Step 2: remove classes not present in y_true
    present = np.unique(y_true)
    keep_idx = [all_classes.index(c) for c in present]
    if len(keep_idx) == 2:
        full = full[:, -1]
    else:
        full = full[:, keep_idx]
        full = full / full.sum(axis=1)[:, None]

    return full


def evaluate(
    Y_prob,
    Y_pred,
    Y_true: pd.DataFrame,
    index="results",
    index_names=None,
    task_type="boolean",
):
    global capture
    col_names = Y_true.columns if task_type == task_types.MULTILABEL.value else ["Y_0"]
    metrics = ["f1", "acc", "precision", "recall", "auc-roc", "mcc"]

    all_cols = [met + "_" + colname for met in metrics for colname in col_names]
    metric_cols = {
        met: [met + "_" + colname for colname in col_names] for met in metrics
    }

    Y_true = np.array(Y_true)

    results = pd.DataFrame(
        columns=all_cols,
        index=pd.MultiIndex.from_tuples([index], names=index_names),
    )
    if task_type == task_types.MULTILABEL.value:
        f1s = apply_to_cols(
            f1_score,
            Y_true,
            Y_pred,
            average="binary",
            zero_division=0,
        )
        accs = apply_to_cols(
            accuracy_score,
            Y_true,
            Y_pred,
        )
        precisions = apply_to_cols(
            precision_score,
            Y_true,
            Y_pred,
            zero_division=0,
            average="binary",
        )
        recalls = apply_to_cols(
            recall_score,
            Y_true,
            Y_pred,
            zero_division=0,
            average="binary",
        )
        rocs = apply_to_cols(
            roc_auc_score,
            Y_true,
            Y_prob,
            average="macro",
        )
        mccs = apply_to_cols(
            matthews_corrcoef,
            Y_true,
            Y_pred,
        )
    else:
        averaging = "binary" if task_type == task_types.BINARY.value else "macro"
        multi_class = "raise" if task_type == task_types.BINARY.value else "ovr"
        f1s = f1_score(
            Y_true,
            Y_pred,
            average=averaging,
            zero_division=0,
        )
        accs = accuracy_score(Y_true, Y_pred)
        precisions = precision_score(Y_true, Y_pred, average=averaging)
        recalls = recall_score(Y_true, Y_pred, average=averaging)
        capture = Y_true, Y_prob, multi_class
        rocs = roc_auc_score(Y_true, Y_prob, average="macro", multi_class=multi_class)
        mccs = matthews_corrcoef(Y_true, Y_pred)

    results.loc[results.index[0], metric_cols["f1"]] = f1s
    results.loc[results.index[0], metric_cols["acc"]] = accs
    results.loc[results.index[0], metric_cols["precision"]] = precisions
    results.loc[results.index[0], metric_cols["recall"]] = recalls
    results.loc[results.index[0], metric_cols["auc-roc"]] = rocs
    results.loc[results.index[0], metric_cols["mcc"]] = mccs

    return results


def predict(model, X, Y_true, trained_classes, all_classes, task_type="boolean"):
    probs = model.predict_proba(X)

    if task_type == task_types.BINARY.value:
        return np.array(probs)[:, 1, None], model.predict(X)

    elif task_type == task_types.MULTICLASS.value:
        return prepare_proba_for_auc(
            np.array(probs), trained_classes, all_classes, Y_true
        ), model.predict(X)

    else:
        for i, table in enumerate(probs):
            if len(table[0]) == 1:
                new_table = [[val[0], 1 - val[0]] for val in table]
                probs[i] = new_table

        return np.array(probs)[:, :, 1].T, model.predict(X)


def grid_search(
    base_model,
    X_train,
    Y_train: pd.DataFrame,
    X_val,
    Y_val,
    X_test,
    Y_test,
    param_grid: dict,
    encoding="encoding",
    model_selection_metric: str = "auc-roc",
    task_type="boolean",
):

    all_classes = list(Y_test[Y_test.columns[0]].unique())

    if task_type != task_types.MULTILABEL.value:
        Y_train = Y_train.values.ravel()
        Y_val = Y_val.values.ravel()
        Y_test = Y_test.values.ravel()

    n_jobs = -1
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []

    best = {
        "model": None,
        "params": None,
        "metric": -np.inf,
    }

    for params in tqdm(
        param_combinations,
        desc=base_model.__name__ + " " + encoding,
        position=4,
        leave=False,
    ):

        if (
            task_type == task_types.MULTILABEL.value
            and base_model == LogisticRegression
        ):
            model = MultiOutputClassifier(base_model(**params), n_jobs=n_jobs)
        else:
            model = base_model(n_jobs=n_jobs, **params)

        model = model.fit(X_train, Y_train)
        # except ValueError as e:
        #     model = DummyClassifier().fit(X_train, Y_train)
        #     print("using dummy")

        prob_train, pred_train = predict(
            model,
            X_train.values,
            Y_train,
            model.classes_,
            all_classes,
            task_type=task_type,
        )
        prob_val, pred_val = predict(
            model,
            X_val.values,
            Y_val,
            model.classes_,
            all_classes,
            task_type=task_type,
        )

        stats_train = evaluate(
            prob_train,
            pred_train,
            Y_train,
            index=["train", encoding, base_model.__name__],
            index_names=["dataset", "encoding", "model"],
            task_type=task_type,
        )
        stats_val = evaluate(
            prob_val,
            pred_val,
            Y_val,
            index=["val", encoding, base_model.__name__],
            index_names=["dataset", "encoding", "model"],
            task_type=task_type,
        )

        for key, value in params.items():
            stats_train.at[stats_train.index[0], key] = value
            stats_val.at[stats_val.index[0], key] = value
        results.append(stats_train)
        results.append(stats_val)

        eval_columns = [
            col for col in stats_val.columns if model_selection_metric in col
        ]
        eval_metric = stats_val[eval_columns].mean(axis=None)

        if eval_metric > best["metric"]:
            best["metric"] = eval_metric
            best["model"] = model
            best["params"] = params

    prob_test, pred_test = predict(
        model,
        X_test.values,
        Y_test,
        model.classes_,
        all_classes,
        task_type=task_type,
    )

    stats_test = evaluate(
        prob_test,
        pred_test,
        Y_test,
        index=["test", encoding, base_model.__name__],
        index_names=["dataset", "encoding", "model"],
        task_type=task_type,
    )
    results.append(stats_test)

    return pd.concat(results), best


# %%
if __name__ == "__main__":

    task_2_name: Dict[str, str] = {
        # Operational outcomes
        "guo_los": "Long LOS",
        "guo_readmission": "30-Day Readmission",
        "guo_icu": "ICU Admission",
        # Anticipating lab test results
        "lab_thrombocytopenia": "Thrombocytopenia",
        "lab_hyperkalemia": "Hyperkalemia",
        "lab_hypoglycemia": "Hypoglycemia",
        "lab_hyponatremia": "Hyponatremia",
        "lab_anemia": "Anemia",
        # Assignment of new diagnoses
        "new_hypertension": "Hypertension",
        "new_hyperlipidemia": "Hyperlipidemia",
        "new_pancan": "Pancreatic Cancer",
        "new_celiac": "Celi ac",
        "new_lupus": "Lupus",
        "new_acutemi": "Acute MI",
        # Anticipating chest x-ray findings
        "chexpert": "Chest X-Ray",
    }

    task_2_value_type: Dict[str, str] = {
        "new_pancan": "boolean",
        "new_celiac": "boolean",
        "new_lupus": "boolean",
        "new_acutemi": "boolean",
        "new_hypertension": "boolean",
        "new_hyperlipidemia": "boolean",
        "guo_los": "boolean",
        "guo_readmission": "boolean",
        "guo_icu": "boolean",
        "lab_thrombocytopenia": "multiclass",
        "lab_hyperkalemia": "multiclass",
        "lab_hypoglycemia": "multiclass",
        "lab_hyponatremia": "multiclass",
        "lab_anemia": "multiclass",
        "chexpert": "multilabel",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        # default="guo_icu",
        default="chexpert",
        help="EHRSHOT task to run classification on",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="categorical_v2",
        help="Encoder to use for classification",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Resample seed for classification",
    )
    parser.add_argument(
        "--chexpert_label",
        type=int,
        default=0,
        help="CheXpert label to use for classification",
    )

    args = parser.parse_args()
    task = args.task

    data_dir = root_dir.joinpath(
        "data", "datasets", "EHRSHOT", task, "encoded", args.encoder
    )
    save_dir = root_dir.joinpath(
        "data",
        "results",
        "ehrshot_stats",
        args.encoder,
        task,
    )

    data_splits = ["train.csv", "val.csv", "test.csv"]
    datas = {}

    for data_split in data_splits:
        data = pd.read_csv(data_dir.joinpath(data_split))
        data = data.iloc[:, 1:]

        if task_2_value_type[task] == task_types.MULTILABEL.value:
            target = (
                data["value"]
                .map(lambda value: list(f"{value:014b}"))
                .apply(pd.Series)
                .add_prefix("Y_")
            )
            print(target.head())
            target = pd.DataFrame(
                target[f"Y_{args.chexpert_label}"].astype(int)
            ).rename(columns={f"Y_{args.chexpert_label}": "Y_0"})
            is_positive = lambda value: value > 0

        elif task_2_value_type[task] == task_types.MULTICLASS.value:
            is_positive = lambda value: value > 0
            target = pd.DataFrame({"Y_0": data["value"]})
        else:
            is_positive = lambda value: value
            target = pd.DataFrame({"Y_0": data["value"].astype(int)})
        data["is_positive"] = target["Y_0"].apply(is_positive)

        data = pd.concat([data, target.astype(int)], axis=1)
        data = data.sample(frac=1, random_state=0).sort_values(by="is_positive")

        datas[data_split] = data

    if task_2_value_type[task] == task_types.MULTILABEL.value:
        task_2_value_type[task] = task_types.BINARY.value

    shuffle_seed = args.shuffle_seed

    data_train_full = (
        datas["train.csv"]
        .sample(frac=1, random_state=shuffle_seed)
        .sort_values(by="is_positive")
    )
    data_val_full = (
        datas["val.csv"]
        .sample(frac=1, random_state=shuffle_seed)
        .sort_values(by="is_positive")
    )
    data_test = datas["test.csv"]

    for shot_size in (
        shot_loop := tqdm(
            [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, -1],
            desc="SHOT SIZE",
            position=0,
            leave=False,
        )
    ):
        shot_loop.set_postfix_str(shot_size)

        results = None
        if shot_size == -1:
            data_train_full = data_train_full.sample(
                frac=1, random_state=shuffle_seed, replace=True
            ).sort_values(by="is_positive")

        shot_save_path = save_dir.joinpath(f"{shot_size}_shot")

        data_train = data_train_full.groupby("is_positive", group_keys=False).head(
            shot_size
        )
        data_val = data_val_full.groupby("is_positive", group_keys=False).head(
            shot_size
        )

        X_cols = [col for col in data_train_full.columns if "X_" in col]
        Y_cols = [col for col in data_train_full.columns if "Y_" in col]

        classifiers = {
            "LogisticRegression": {
                "model": LogisticRegression,
                "param_grid": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["saga"],
                    "class_weight": ["balanced"],
                    "max_iter": [200],
                    "random_state": [shuffle_seed],
                },
            },
            "RandomForest": {
                "model": RandomForestClassifier,
                "param_grid": {
                    "n_estimators": [10, 25, 50, 100, 250],
                    "max_depth": [2, 5, 10, None],
                    "class_weight": ["balanced"],
                    "random_state": [shuffle_seed],
                },
            },
        }

        X_train = data_train[X_cols]
        Y_train = data_train[Y_cols]

        X_val = data_val[X_cols]
        Y_val = data_val[Y_cols]

        X_test = data_test[X_cols]
        Y_test = data_test[Y_cols]

        for classifier_name, classifier in classifiers.items():
            # if classifier_name == "LogisticRegression" and task == "chexpert":
            #     continue

            res, model = grid_search(
                classifier["model"],
                X_train,
                Y_train,
                X_val,
                Y_val,
                X_test,
                Y_test,
                classifier["param_grid"],
                args.encoder,
                task_type=task_2_value_type[task],
            )
            # except ValueError as e:
            #     print(
            #         "valueerror at shot size ",
            #         shot_size,
            #         " and shuffle seed ",
            #         shuffle_seed,
            #         " discarding",
            #     )
            #     continue

            res["resample_seed"] = shuffle_seed
            res["shot_size"] = shot_size

            results = pd.concat([results, res])

            classifier_save_dir = shot_save_path.joinpath(
                "classifiers",
                classifier_name,
                f"shuffle_seed_{shuffle_seed}",
                f"shot_size_{shot_size}",
            )
            if args.task == "chexpert":
                classifier_save_dir = classifier_save_dir.joinpath(
                    f"chexpert_label_{args.chexpert_label}"
                )

            classifier_save_dir.mkdir(parents=True, exist_ok=True)
            with open(classifier_save_dir.joinpath(f"best_model.pkl"), "wb") as file:
                pickle.dump(model["model"], file)

            with open(classifier_save_dir.joinpath(f"params.json"), "w") as file:
                json.dump(model["params"], file, sort_keys=True, indent=4)

        if results is not None:
            results_save_path = shot_save_path.joinpath(
                f"stats_shuffle_seed_{shuffle_seed}.csv"
            )
            if args.task == "chexpert":
                results_save_path = shot_save_path.joinpath(
                    f"chexpert_label_{args.chexpert_label}",
                    f"stats_shuffle_seed_{shuffle_seed}.csv",
                )
            results_save_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(results_save_path)

# save_dir.joinpath("finished_indicator").touch()
# %%
