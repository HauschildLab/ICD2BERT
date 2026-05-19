# %%
from encoder import (
    BertEncoder,
    Pat2VecEncoder,
    BehrtEncoder,
    CategoricalEncoder,
    CategoricalEncoder_v2,
    CLMBrEncoder,
)
import sys
import pandas as pd
import os
from const import root_dir
import torch
import argparse

# sys.argv = [sys.argv[0], "--dataset", "mimic_iv_2.2", "--model", "CLMBR"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="mimic_iv_2.2",
    help="Dataset to encode",
)

parser.add_argument(
    "--variation",
    type=str,
    default="",
    help="Dataset variation to encode",
)

parser.add_argument(
    "--model", type=str, default="categorical_v2", help="Model to use for encoding"
)

args = parser.parse_args()

print(f"{torch.cuda.is_available()=}")

models_path = root_dir.joinpath("data", "models")

model_name = args.model
# %%

is_behrt = False
model = None
if model_name == "CLMBR":
    model = CLMBrEncoder(device="cuda")
elif "beHrt" in model_name or "med-bert" in model_name:
    is_behrt = True
    model = BehrtEncoder(models_path.joinpath(model_name, "checkpoint-55700"))

elif "pat2vec_base" in model_name:
    model = Pat2VecEncoder(str(models_path.joinpath(model_name, "pat2vec_dim10.model")))
elif "bert_" in model_name:
    checkpoints = [
        p.name
        for p in models_path.joinpath(model_name).iterdir()
        if "checkpoint" in p.name
    ]
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    encoder_path = models_path.joinpath(model_name, checkpoints[-1])
    model = BertEncoder(encoder_path)
elif "pat2vec_" in model_name:
    encoder_path = models_path.joinpath(model_name, "best.model")
    model = Pat2VecEncoder(str(encoder_path))
elif "categorical_v2" in model_name:
    model = CategoricalEncoder_v2(models_path.joinpath("categorical"))
elif "categorical" in model_name:
    model = CategoricalEncoder(models_path.joinpath(model_name, "icd_categories.csv"))

if args.dataset == "mimic_iv_2.2":
    if (
        "ki_thrust" in model_name
        or "pat2vec_base" in model_name
        or "_sep_" in model_name
        or "categorical" in model_name
        or is_behrt
    ):
        dataset_names = ["mimic_iv_2.2", "mimic_iv_2.2_full"]
    else:
        dataset_names = ["_".join(model_name.split("_")[2:])]
        dataset_names.append(dataset_names[-1] + "_full")
    if model_name == "CLMBR":
        dataset_names = ["mimic_iv_2.2_EHRSHOT", "mimic_iv_2.2_EHRSHOT_full"]
    if dataset_names[0] == "":
        print("using default fallback dataset")
        dataset_names = ["mimic_iv_2.2", "mimic_iv_2.2_full"]


if args.dataset == "EHRSHOT":
    dataset_names = [args.variation]

print(model_name)
print(dataset_names, "\n" * 3)

# %%
for dataset_name in dataset_names:
    datasets = ["train.csv", "val.csv", "test.csv"]
    for part in datasets:

        if "mimic" in args.dataset:
            savepath = root_dir.joinpath(
                "data",
                "datasets",
                dataset_name,
                "encoded",
                "_".join(
                    model_name.split("_")[: 2 if "ki_thrust" not in model_name else 4]
                ),
            )
            datapath = root_dir.joinpath(
                "data",
                "datasets",
                dataset_name,
                part,
            )
        else:
            savepath = root_dir.joinpath(
                "data",
                "datasets",
                "EHRSHOT",
                dataset_name,
                "encoded",
                model_name,
            )
            datapath = root_dir.joinpath(
                "data",
                "datasets",
                "EHRSHOT",
                dataset_name,
                part,
            )

        # if savepath.joinpath(part).exists():
        #     print("already encoded, skipping")
        #     continue

        print(datapath)

        data = pd.read_csv(datapath)

        icd_col = "previous_diagnoses" if "mimic" in args.dataset else "icd_codes"
        seg_id_alt_col = "alt_seg_ids" if "mimic" in args.dataset else "seg_ids_alt"

        data[icd_col] = data[icd_col].fillna("")

        if model_name == "CLMBR":
            encodings = model.encode(data)
        elif not is_behrt:
            encodings = model.encode(list(data[icd_col].values))
        else:
            encodings = model.encode(
                data[icd_col],
                data["age_ids"],
                data["pos_ids"],
                data["seg_ids"],
                data[seg_id_alt_col],
            )

        data_with_encodings = pd.concat(
            [data, encodings.add_prefix("X_" + model_name + "_")], axis=1
        )

        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        data_with_encodings.to_csv(savepath.joinpath(part))

# %%
