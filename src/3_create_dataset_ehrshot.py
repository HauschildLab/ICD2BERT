# %%
import pandas as pd
from const import root_dir
import json

data_path = root_dir.joinpath("data", "EHRSHOT_ASSETS", "data", "ehrshot.csv")
data = pd.read_csv(data_path, index_col=0)
data["start"] = pd.to_datetime(data["start"])
data["end"] = pd.to_datetime(data["end"])
data_icd = data[data["code"].map(lambda code: "ICD" in code.upper())].copy()

mimic_vocab_path = root_dir.joinpath("data", "datasets", "mimic_iv_2.2", "vocab.csv")
mimic_vocab = pd.read_csv(mimic_vocab_path, header=None)
# %%


def other_code(code):
    return "ICD" not in code


def icd09proc(code):
    return "ICD9Proc" in code


def icd10diag(code):
    return "ICDO3" in code


def icd10proc(code):
    return "ICD10PCS" in code


code_filter_funks = [
    ("other", other_code),
    ("ICD10Proc", icd10proc),
    ("ICD10DIAG", icd10diag),
    ("ICD9Proc", icd09proc),
]

for code_type, filter_funk in code_filter_funks:
    count = data["code"].map(filter_funk).sum()
    print(code_type, count)


# %%
icd_lookup = set(mimic_vocab[0])


def ehrshot_to_mimic_code(code: str):

    global icd_lookup
    if "ICD9Proc" in code:
        code = "p_09_" + code.split("/")[-1].replace(".", "")
        if code in icd_lookup:
            return code

    if "ICDO3" in code:
        split_char = "-" if "-" in code else "/"
        code = "d_10_" + code.split(split_char)[-1].replace(".", "")
        if code in icd_lookup:
            return code
        if code[:-1] in icd_lookup:
            return code[:-1]

    if "ICD10PCS" in code:
        code = "p_10_" + code.split("/")[-1].replace(".", "")
        if code in icd_lookup:
            return code

    return ""


data_icd["mimic_code"] = data_icd["code"].map(lambda code: ehrshot_to_mimic_code(code))


# %%

should_be_done = data_icd["code"].map(lambda code: "ICD10PCS" in code)
is_done = data_icd["mimic_code"] != ""

data_missing = data_icd[should_be_done & ~is_done]
data_missing

# %%
code_types = data_icd["code"].map(lambda code: code.split("/")[0]).unique()
code_types

data_rem = data_icd[data_icd["code"].map(lambda code: code_types[2] in code)]
data_rem

data_icd = data_icd[is_done]
# %%
patient_metadata_path = root_dir.joinpath("data", "EHRSHOT_ASSETS", "person.csv")
patient_metadata = pd.read_csv(patient_metadata_path)
patient_metadata["birth_DATETIME"] = pd.to_datetime(patient_metadata["birth_DATETIME"])
patient_metadata["patient_id"] = patient_metadata["person_id"].astype(int)
patient_date_of_birth = patient_metadata[["patient_id", "birth_DATETIME"]]

# data_icd = data_icd.drop(columns="birth_DATETIME", errors="ignore")

data_icd = pd.merge(
    left=data_icd,
    right=patient_date_of_birth,
    on="patient_id",
    how="left",
)

data_icd = data_icd.drop(data_icd[data_icd["birth_DATETIME"].isna()].index)

data_icd["age"] = data_icd["start"] - data_icd["birth_DATETIME"]
data_icd["age"] = data_icd["age"].map(
    lambda timedelta: str(int(timedelta.days / 365.25))
)

data_icd = data_icd.sort_values(by="start")
data_icd["visit_id"] = data_icd["visit_id"].fillna(-1)

# %%

patient_split_path = root_dir.joinpath(
    "data", "EHRSHOT_ASSETS", "splits", "person_id_map.csv"
)
patient_split = pd.read_csv(patient_split_path)

splits = [
    {
        "split_path": f"{split}.csv",
        "patient_ids": set(
            patient_split[patient_split["split"] == split]["omop_person_id"]
        ),
    }
    for split in ["train", "val", "test"]
]


def make_pos_ids(icd_codes):
    words = icd_codes.split()
    pos_ids = list(range(1, len(words) + 1))
    return " ".join(str(x) for x in pos_ids)


def make_seg_ids(icd_codes):
    words = icd_codes.split()
    seg_counter = 1
    seg_ids = []

    for word in words:
        seg_ids.append(seg_counter)
        if word == "[SEP]":
            seg_counter += 1

    return " ".join(str(x) for x in seg_ids)


def make_seg_ids_alt(icd_codes):
    words = icd_codes.split()
    seg_counter = 0
    seg_ids = []

    for word in words:
        seg_ids.append(seg_counter)
        if word == "[SEP]":
            seg_counter = (seg_counter + 1) % 2

    return " ".join(str(x) for x in seg_ids)


# # %%
# has_data_count = 0
# no_data_count = 0

# with open(
#     root_dir.joinpath(
#         "data", "EHRSHOT_ASSETS", "benchmark", "guo_icu", "all_shots_data.json"
#     ),
#     "r",
# ) as f:
#     data_shots = json.load(f)

#     assert isinstance(data_shots, dict)

#     for benchmark_name, benchmark in data_shots.items():
#         for shot_size, shot in benchmark.items():
#             if shot_size == "-1":
#                 continue
#             for repetition_id, repetition in shot.items():
#                 for data_split in ["train"]:
#                     patient_ids = repetition[f"patient_ids_{data_split}_k"]
#                     patient_times = repetition[f"label_times_{data_split}_k"]

#                 for id, time in zip(patient_ids, patient_times):
#                     time = pd.to_datetime(time)

#                     num_samples = len(
#                         data_icd[
#                             (data_icd["start"] <= time) & (data_icd["patient_id"] == id)
#                         ]
#                     )
#                     if num_samples > 0:
#                         has_data_count += 1
#                     else:
#                         no_data_count += 1

# print(
#     f"{has_data_count=}, {no_data_count=}, data_retained={100*has_data_count/(has_data_count + no_data_count):.1f}%"
# )


# %%

benchmarks_path = root_dir.joinpath("data", "EHRSHOT_ASSETS", "benchmark")

for task in benchmarks_path.iterdir():
    print(task)
    if task.name in ["new_hyperlipidemia", "guo_readmission", "lab_thrombocytopenia"]:
        print("skipping")
        continue

    task_label_path = task.joinpath("labeled_patients.csv")
    if not task_label_path.exists():
        continue

    task_labels = pd.read_csv(task_label_path)
    task_labels["prediction_time"] = pd.to_datetime(task_labels["prediction_time"])

    drop_indexes = []

    for index, row in task_labels.iterrows():
        patient_id = row["patient_id"]
        prediction_time = row["prediction_time"]

        relevant_icd_data = data_icd[
            (data_icd["patient_id"] == patient_id)
            & (data_icd["start"] <= prediction_time)
        ].copy()

        if len(relevant_icd_data) == 0:
            drop_indexes.append(index)
            continue

        admission_change = (
            relevant_icd_data["visit_id"] != relevant_icd_data["visit_id"].shift()
        )

        relevant_icd_data.loc[admission_change, "mimic_code"] = (
            "[SEP] " + (relevant_icd_data.loc[admission_change, "mimic_code"])
        )

        relevant_icd_data.loc[admission_change, "age"] = (
            relevant_icd_data.loc[admission_change, "age"]
            + " "
            + relevant_icd_data.loc[admission_change, "age"]
        )

        icd_codes = " ".join(relevant_icd_data["mimic_code"])[6:].strip()
        timestamps = " ".join(relevant_icd_data["start"].astype(str))
        og_codes = " ".join(relevant_icd_data["code"])
        age_ids = " ".join(" ".join(list(relevant_icd_data["age"])).split(" ")[1:])
        pos_ids = make_pos_ids(icd_codes)
        seg_ids = make_seg_ids(icd_codes)
        seg_ids_alt = make_seg_ids_alt(icd_codes)

        task_labels.loc[index, "icd_codes"] = icd_codes
        task_labels.loc[index, "timestamps"] = timestamps
        task_labels.loc[index, "og_codes"] = og_codes
        task_labels.loc[index, "age_ids"] = age_ids
        task_labels.loc[index, "pos_ids"] = pos_ids
        task_labels.loc[index, "seg_ids"] = seg_ids
        task_labels.loc[index, "seg_ids_alt"] = seg_ids_alt

    task_labels.drop(index=drop_indexes, inplace=True)
    task_labels = pd.merge(
        left=task_labels,
        right=patient_metadata[["patient_id", "birth_DATETIME"]],
        on="patient_id",
        how="left",
    )

    for split in splits:
        savepath = root_dir.joinpath("data", "datasets", "EHRSHOT", task.name)
        savepath.mkdir(parents=True, exist_ok=True)

        split_mask = task_labels["patient_id"].map(
            lambda patient: patient in split["patient_ids"]
        )

        task_labels[split_mask].to_csv(savepath.joinpath(split["split_path"]))

# %%
