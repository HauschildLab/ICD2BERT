# %%
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from const import root_dir


def to_uid(row, code_type="d"):
    return f'{code_type}_{row["icd_version"]:02d}_{row["icd_code"]} '


def has_rehosp(admissions):
    admissions = admissions.sort_values("admittime")  # just to be safe
    if len(admissions) < 2:
        return 0

    second_last_discharge = admissions.iloc[-2]["dischtime"]
    last_admission = admissions.iloc[-1]["admittime"]
    return 1 * ((last_admission - second_last_discharge).days <= 30)


mimic_version = "mimic_iv_2.2"
base_mimic_path = root_dir.joinpath("data", mimic_version.replace("_", "-"), "hosp")

data_procedures = pd.read_csv(
    base_mimic_path.joinpath("procedures_icd.csv.gz")
).sort_values(by="chartdate")
data_diagnosis = pd.read_csv(
    base_mimic_path.joinpath("diagnoses_icd.csv.gz")
).sort_values(by="seq_num")
data_admissions = pd.read_csv(base_mimic_path.joinpath("admissions.csv.gz"))

data_procedures["uid"] = data_procedures.apply(to_uid, code_type="p", axis=1)
data_diagnosis["uid"] = data_diagnosis.apply(to_uid, code_type="d", axis=1)


admissions_procedures = data_procedures[["hadm_id", "uid"]].groupby(by="hadm_id").sum()
admissions_diagnosis = data_diagnosis[["hadm_id", "uid"]].groupby(by="hadm_id").sum()

admissions_all = pd.merge(
    left=data_admissions,
    right=admissions_procedures.add_suffix("_proc"),
    how="left",
    left_on="hadm_id",
    right_index=True,
)

admissions_all = pd.merge(
    left=admissions_all,
    right=admissions_diagnosis.add_suffix("_diag"),
    how="left",
    left_on="hadm_id",
    right_index=True,
)

admissions_all["admittime"] = pd.to_datetime(admissions_all["admittime"])
admissions_all["dischtime"] = pd.to_datetime(admissions_all["dischtime"])

admissions_all["uid_proc"] = admissions_all["uid_proc"].fillna("")
admissions_all["uid_diag"] = admissions_all["uid_diag"].fillna("")

admissions_all["all_uids"] = admissions_all["uid_proc"] + admissions_all["uid_diag"]
admissions_all = admissions_all[admissions_all["all_uids"].map(lambda x: len(x) > 0)]

# print(admissions_all["all_uids"].map(lambda x: len(x)).min())
# print("bal", admissions_all["all_uids"][admissions_all["all_uids"].map(lambda x: len(x)).argmin()])

admissions_all = admissions_all.drop(columns=["uid_proc", "uid_diag"])

admissions_all = admissions_all.sort_values(by=["subject_id", "admittime"])

rehospitalizations = (
    admissions_all.groupby("subject_id")[["subject_id", "admittime", "dischtime"]]
    .apply(has_rehosp)
    .reset_index(name="rehosp_30")
)

admissions_all = pd.merge(
    left=admissions_all, right=rehospitalizations, on="subject_id", how="left"
)

admissions_all = admissions_all.sort_values(by="dischtime")
has_multiple_admissions = admissions_all["subject_id"].duplicated(keep=False)
is_last_admission = ~admissions_all["subject_id"].duplicated(keep="last")
admissions_previous = admissions_all[
    has_multiple_admissions & ~is_last_admission
].copy()
admissions_last = admissions_all[has_multiple_admissions & is_last_admission].copy()

admissions_previous["all_uids"] = admissions_previous["all_uids"] + "[SEP] "

data_patient = pd.read_csv(base_mimic_path.joinpath("patients.csv.gz"))
data_patient["anchor_year"] = pd.to_datetime(data_patient["anchor_year"], format="%Y")

admissions_previous_age = pd.merge(
    left=admissions_previous,
    right=data_patient[["subject_id", "anchor_year", "anchor_age"]],
    how="left",
    on="subject_id",
)

admissions_previous_age["age"] = (
    admissions_previous_age["anchor_age"]
    + (
        admissions_previous_age["admittime"] - admissions_previous_age["anchor_year"]
    ).dt.days
    / 365.25
).astype(int)

# admissions_previous_age["age_exact"] = (
#     admissions_previous_age["anchor_age"]
#     + (
#         admissions_previous_age["admittime"] - admissions_previous_age["anchor_year"]
#     ).dt.days
#     / 365.25
# )
# admissions_previous_age['birth_DATETIME'] = admissions_previous_age['anchor_year'] - pd.to_timedelta((365.25 * admissions_previous_age['anchor_age']).astype(int), unit='D')
admissions_previous_age["birth_DATETIME"] = admissions_previous_age.apply(
    lambda row: pd.Timestamp(
        year=row["anchor_year"].year - int(row["anchor_age"]),
        month=1,
        day=1,
    ),
    axis=1,
)

admissions_previous_age["age_ids"] = admissions_previous_age.apply(
    lambda row: (str(row["age"]) + " ") * len(row["all_uids"].split()), axis=1
)

admissions_previous_age["timestamps"] = admissions_previous_age.apply(
    lambda row: (str(row["admittime"]) + " ") * len(row["all_uids"].split()), axis=1
)

icds_previous = (
    admissions_previous_age[["subject_id", "all_uids", "age_ids", "timestamps"]]
    .astype(str)
    .groupby(by="subject_id")
    .sum()
)

icds_previous.index = icds_previous.index.astype(int)

# icds_previous.rename(columns={"admittime_expanded": "timestamps"}, inplace=True)

exact_age = (
    admissions_previous_age[["subject_id", "birth_DATETIME"]]
    .groupby(by="subject_id")
    .min()
)

icds_previous = pd.merge(
    left=icds_previous,
    right=exact_age,
    how="left",
    left_index=True,
    right_index=True,
)

admissions_split = pd.merge(
    left=admissions_last,
    right=icds_previous.rename(columns={"all_uids": "previous_uids"}),
    how="left",
    left_on="subject_id",
    right_index=True,
)

admissions_with_tasks = pd.merge(
    left=admissions_all,
    right=admissions_split[
        ["hadm_id", "previous_uids", "age_ids", "birth_DATETIME", "timestamps"]
    ],
    how="left",
    left_on="hadm_id",
    right_on="hadm_id",
)


def make_pos_ids(row):
    words = row["previous_uids"].split()
    pos_ids = list(range(1, len(words) + 1))
    return " ".join(str(x) for x in pos_ids)


def make_seg_ids(row):
    words = row["previous_uids"].split()
    seg_counter = 1
    seg_ids = []

    for word in words:
        seg_ids.append(seg_counter)
        if word == "[SEP]":
            seg_counter += 1

    return " ".join(str(x) for x in seg_ids)


def make_alt_seg_ids(row):
    words = row["previous_uids"].split()
    seg_counter = 0
    seg_ids = []

    for word in words:
        seg_ids.append(seg_counter)
        if word == "[SEP]":
            seg_counter = (seg_counter + 1) % 2

    return " ".join(str(x) for x in seg_ids)


admissions_with_tasks = admissions_with_tasks[
    ~admissions_with_tasks["previous_uids"].isna()
].copy()
admissions_with_tasks["pos_ids"] = admissions_with_tasks.apply(make_pos_ids, axis=1)
admissions_with_tasks["seg_ids"] = admissions_with_tasks.apply(make_seg_ids, axis=1)
admissions_with_tasks["alt_seg_ids"] = admissions_with_tasks.apply(
    make_alt_seg_ids, axis=1
)

mlb_classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mlb = MultiLabelBinarizer().fit(mlb_classes)

future_bin = admissions_with_tasks[
    ~admissions_with_tasks["previous_uids"].isna()
].copy()

future_bin["class"] = future_bin["all_uids"].map(
    lambda x: set(
        map(lambda z: z[5], filter(lambda y: y.startswith("d_10"), x.split(" ")))
    )
)
future_classes = pd.DataFrame(
    mlb.transform(future_bin["class"]), columns=list(mlb_classes), dtype=int
)
future_classes = future_classes.loc[:, future_classes.mean(axis=0) != 0]

future_all = pd.concat([future_bin.reset_index(drop=True), future_classes], axis=1)

full_dataset = pd.merge(
    left=admissions_with_tasks,
    right=future_all[
        ["hadm_id"] + [col for col in future_all.columns if col in mlb_classes]
    ],
    how="left",
    on="hadm_id",
)
full_dataset = full_dataset.drop(columns=["all_uids"])
full_dataset = full_dataset[~full_dataset["previous_uids"].isna()]
full_dataset["previous_uids"] = full_dataset["previous_uids"].map(lambda x: x[:-7])

task_columns = [
    "rehosp_30",
    "hospital_expire_flag",
    "previous_uids",
    "age_ids",
    "pos_ids",
    "seg_ids",
    "alt_seg_ids",
    "birth_DATETIME",
    "timestamps",
] + mlb_classes
reordered_dataset = full_dataset[
    [col for col in full_dataset.columns if col not in task_columns]
].copy()
reordered_dataset["previous_diagnoses"] = full_dataset["previous_uids"]
reordered_dataset["timestamps"] = full_dataset["timestamps"]
reordered_dataset["birth_DATETIME"] = full_dataset["birth_DATETIME"]
reordered_dataset["age_ids"] = full_dataset["age_ids"]
reordered_dataset["pos_ids"] = full_dataset["pos_ids"]
reordered_dataset["seg_ids"] = full_dataset["seg_ids"]
reordered_dataset["alt_seg_ids"] = full_dataset["alt_seg_ids"]

reordered_dataset["mortality"] = full_dataset["hospital_expire_flag"]
reordered_dataset["rehospitalization"] = full_dataset["rehosp_30"]

reordered_dataset = pd.concat(
    [
        reordered_dataset,
        full_dataset[
            [col for col in full_dataset.columns if col in mlb_classes]
        ].astype(int),
    ],
    axis=1,
)

train, val_test = train_test_split(reordered_dataset, test_size=0.3, random_state=0)
val, test = train_test_split(val_test, test_size=0.5, random_state=0)

for affix in ["", "_full"]:
    savepath = root_dir.joinpath("data", "datasets", mimic_version + affix)
    savepath.mkdir(parents=True, exist_ok=True)

    train.to_csv(savepath.joinpath("train.csv"), index=False)
    val.to_csv(savepath.joinpath("val.csv"), index=False)
    test.to_csv(savepath.joinpath("test.csv"), index=False)

    codes_procedures = pd.read_csv(base_mimic_path.joinpath("d_icd_procedures.csv.gz"))
    codes_diagnosis = pd.read_csv(base_mimic_path.joinpath("d_icd_diagnoses.csv.gz"))

    codes_procedures["uid"] = codes_procedures.apply(to_uid, code_type="p", axis=1)
    codes_diagnosis["uid"] = codes_diagnosis.apply(to_uid, code_type="d", axis=1)
    codes_all = pd.concat([codes_diagnosis["uid"], codes_procedures["uid"]])

    codes_all = codes_all.map(lambda x: x.strip())
    codes_all.to_csv(savepath.joinpath("vocab.csv"), index=False, header=False)


# %%
