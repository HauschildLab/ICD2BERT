# %%
import pandas as pd
from const import root_dir
from typing import Dict
from matplotlib import pyplot as plt
import matplotlib.axes
import numpy as np

eval_stat = "auc-roc"
# eval_model = "LogisticRegression"
eval_model = "RandomForestClassifier"

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


name_2_type: Dict[str, str] = {
    # Operational outcomes
    "Long LOS": "Operational Outcomes",
    "30-Day Readmission": "Operational Outcomes",
    "ICU Admission": "Operational Outcomes",
    # Anticipating lab test results
    "Thrombocytopenia": "Anticipating Lab Test Results",
    "Hyperkalemia": "Anticipating Lab Test Results",
    "Hypoglycemia": "Anticipating Lab Test Results",
    "Hyponatremia": "Anticipating Lab Test Results",
    "Anemia": "Anticipating Lab Test Results",
    # Assignment of new diagnoses
    "Hypertension": "Assignment of New Diagnoses",
    "Hyperlipidemia": "Assignment of New Diagnoses",
    "Pancreatic Cancer": "Assignment of New Diagnoses",
    "Celi ac": "Assignment of New Diagnoses",
    "Lupus": "Assignment of New Diagnoses",
    "Acute MI": "Assignment of New Diagnoses",
    # Anticipating chest x-ray findings
    "Chest X-Ray": "Anticipating Chest X-ray Findings",
}

figures_path = "../data/results/tables_and_figures"
stats = ["f1", "acc", "precision", "recall", "auc-roc", "mcc"]
shot_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, -1]


results_path = root_dir.joinpath("data", "results")

ehr_shot_path = results_path.joinpath("ehrshot_stats")

models_and_colors = [
    ["bert_240_mimic_iv_2.2", "#1E88E5"],
    ["beHrt_240", "#D81B60"],
    ["med-bert_240", "#FFC107"],
    ["pat2vec_240", "#004D40"],
    ["categorical_v2", "#808080"],
    ["CLMBR", "#5E0C35"],
]

model_fancy_and_colors = [
    ["bert", "#1E88E5"],
    ["beHrt", "#D81B60"],
    ["med-bert", "#FFC107"],
    ["pat2vec", "#004D40"],
    ["categorical", "#808080"],
    ["CLMBR", "#5E0C35"],
]
all_results = {}

for model, model_color in models_and_colors:
    task_paths = list(ehr_shot_path.joinpath(model).iterdir())

    def get_results(task_path, task_index=0, chexpert=False):
        results = {}
        if chexpert:
            chexpert_folder = f"chexpert_label_{task_index}"
            task_index = 0

        for shot_size in shot_sizes:
            data_path = task_path.joinpath(f"{shot_size}_shot")
            if chexpert:
                data_path = data_path.joinpath(chexpert_folder)
            shot_results = []
            for shuffle_seed in range(5):
                data = pd.read_csv(
                    data_path.joinpath(f"stats_shuffle_seed_{shuffle_seed}.csv")
                )

                test_results = data[
                    (data["dataset"] == "test") & (data["model"] == eval_model)
                ]
                performance_value = float(
                    test_results[f"{eval_stat}_Y_{task_index}"].values[0]
                )
                shot_results.append(performance_value)

            results[str(shot_size)] = {
                "all": shot_results,
                "mean": sum(shot_results) / len(shot_results),
            }
        return results

    results = {}

    for task_path in task_paths:
        if "chexpert" == task_path.name:
            results[task_path.name] = {}
            for task_index in range(14):
                results[task_path.name][f"Task {task_index}"] = get_results(
                    task_path, task_index=task_index, chexpert=True
                )
        else:
            # if model in ["categorical_v2", "CLMBR"]:
            #     results[task_2_name[task_path.name]] = get_results(task_path)
            # else:
            results[task_path.name] = get_results(task_path)

    all_results[model] = {"color": model_color, "results": results}


# %%

task_type_2_ax: Dict[str, str] = {
    "Operational Outcomes": (0, 0),
    "Anticipating Lab Test Results": (0, 1),
    "Assignment of New Diagnoses": (1, 0),
    "Anticipating Chest X-ray Findings": (1, 1),
}

fig, axs = plt.subplots(2, 2, figsize=(12, 12))


def get_shot_stats(shot_stats_raw: Dict):
    shot_stats = []
    for _, stats in shot_stats_raw.items():
        shot_stats.append(stats["mean"])
    return shot_stats


for model_name, model_stats in all_results.items():
    # if model_name != "categorical_v2":
    #     continue

    plot_lines = {key: [] for key in task_type_2_ax.keys()}
    for task, task_results in model_stats["results"].items():
        if "chexpert" == task:
            for task_id, task_id_results in task_results.items():
                plot_lines[name_2_type[task_2_name[task]]].append(
                    get_shot_stats(task_id_results)
                )
        else:
            plot_lines[name_2_type[task_2_name[task]]].append(
                get_shot_stats(task_results)
            )

    for task_type, (ax_x, ax_y) in task_type_2_ax.items():
        # if task_type == "Anticipating Chest X-ray Findings" and model_name in [
        #     "categorical_v2",
        #     "CLMBR",
        # ]:
        #     continue
        current_ax = axs[ax_x, ax_y]
        print(task_type, ax_x, ax_y)
        assert isinstance(current_ax, matplotlib.axes.Subplot)
        current_ax.set_title(task_type)
        current_ax.set_xlabel("# of Train Examples per Class")
        current_ax.set_ylabel(eval_stat)
        X = shot_sizes.copy()
        X[-1] = 200
        current_ax.set_xscale("log")
        x_labels = [str(size) if size != -1 else "All" for size in shot_sizes]
        current_ax.set_xticks(X, x_labels)
        Ys = np.array(plot_lines[task_type]).T
        print(task, Ys.shape)
        current_ax.plot(X, Ys, alpha=0.25, color=model_stats["color"])
        current_ax.plot(X, Ys.mean(axis=1), color=model_stats["color"], marker="o")

        fig.legend()

for name, color in model_fancy_and_colors:
    plt.scatter([], [], color=color, label=name)

fig.legend(
    loc="lower center",
    ncol=len(models_and_colors),
    # handletextpad=1.0,
    # columnspacing=1.5,
    frameon=False,
)

plt.savefig(f"{figures_path}/ehrshot_4_plots_{eval_stat}_{eval_model}.pdf")
plt.savefig(f"{figures_path}/ehrshot_4_plots_{eval_stat}_{eval_model}.png")
plt.show()


# %%

mimic_ki_thrust_results = None

mimic_ki_thrust_path = results_path.joinpath("classification_stats")

for dataset in ["mimic", "ki_thrust"]:
    for task in ["future_disease", "mortality", "rehospitalization"]:
        task_path = mimic_ki_thrust_path.joinpath(dataset, task)
        for dataset_variation in task_path.iterdir():
            for encoder in dataset_variation.iterdir():
                for resample_seed in range(10):
                    try:
                        raw_data = pd.read_csv(
                            encoder.joinpath(f"stats_resample_seed_{resample_seed}.csv")
                        )
                    except FileNotFoundError:
                        if encoder.name in ["categorical_v2", "CLMBR"]:
                            continue
                        raw_data = pd.read_csv(
                            encoder.joinpath(
                                f"stats_resample_seed_{resample_seed + 10}.csv"
                            )
                        )

                    data = raw_data[
                        (raw_data["model"] == eval_model)
                        & (raw_data["dataset"] == "test")
                    ]
                    classification_result = float(
                        data[[col for col in data.columns if eval_stat in col]].mean(
                            axis=None
                        )
                    )

                    res_dict = {
                        "dataset": [dataset],
                        "task": [task],
                        "dataset_variation": [dataset_variation.name],
                        "encoder": [encoder.name],
                        "resample_seed": [resample_seed],
                        eval_stat: [classification_result],
                    }

                    res_df = pd.DataFrame.from_dict(
                        res_dict,
                    )

                    mimic_ki_thrust_results = pd.concat(
                        [mimic_ki_thrust_results, res_df]
                    )

# %%
groups = mimic_ki_thrust_results.groupby(
    by=["dataset", "task", "dataset_variation", "encoder"]
)
means = groups[eval_stat].transform("mean")
stds = groups[eval_stat].transform("std")

mimic_ki_thrust_results["mean"] = means
mimic_ki_thrust_results["std"] = stds

mimic_ki_thrust_results["raw_metric"] = mimic_ki_thrust_results[eval_stat]
mimic_ki_thrust_results["display"] = mimic_ki_thrust_results.apply(
    lambda row: f"{row['mean']:.2f} ({row['std']:.2f})", axis=1
)
# %%
results_table = mimic_ki_thrust_results.copy()
results_table = results_table[results_table["resample_seed"] == 0]

results_table = results_table.set_index(["dataset", "dataset_variation", "encoder"])

results_table

table: pd.DataFrame = None

for task in results_table["task"].unique():
    task_table = results_table[results_table["task"] == task].copy()
    task_table[task] = task_table["display"]

    table = pd.concat([table, task_table[[task]]], axis=1)

# %%


def encoder_name_to_stats(name: str) -> dict:
    res = {}
    if "240" in name or name == "categorical" or name == "bert_base":
        res["dimensionality"] = 240
    elif "768" in name or name == "CLMBR":
        res["dimensionality"] = 768
    elif "636" in name:
        res["dimensionality"] = 636
    elif "640" in name or name == "categorical_v2":
        res["dimensionality"] = 640
    elif "base" in name:
        res["dimensionality"] = 10

    if "med-bert" in name:
        res["model type"] = "Med-BERT"
    elif "CLMBR" in name:
        res["model type"] = "CLMBR"
    elif "bert" in name:
        res["model type"] = "BERT"
    elif "beHrt" in name:
        res["model type"] = "BEHRT"
    elif "pat2vec" in name:
        res["model type"] = "Pat2Vec"
    elif "categorical" in name:
        res["model type"] = "Categorical"

    if "ki-thrust" in name or "ki_thrust" in name:
        res["training dataset"] = "Ki-Thrust"
    elif name == "pat2vec_base":
        res["training dataset"] = "Pat2Vec"
    elif name == "CLMBR":
        res["training dataset"] = "CLMBR"
    elif ("base" in name or "categorical" in name) and "bert" not in name:
        res["training dataset"] = "None"
    else:
        res["training dataset"] = "MIMIC-IV"

    return res


deconstructed_encoder_names = (
    table.reset_index()["encoder"].apply(encoder_name_to_stats).apply(pd.Series)
)

table_decon = pd.concat([table.reset_index(), deconstructed_encoder_names], axis=1)
table_decon["evaluation dataset"] = table_decon["dataset"]

# %% mimic table

mimic_results = table_decon[
    (table_decon["evaluation dataset"] == "mimic")
    & (
        (
            (table_decon["dataset_variation"] == "mimic_iv_2.2_full")
            | (table_decon["dataset_variation"] == "mimic_iv_2.2_EHRSHOT_full")
        )
        | (table_decon["dataset_variation"] == "ki_thrust")
    )
]
mimic_results = mimic_results[mimic_results["encoder"] != "bert_base"]

mimic_results = mimic_results[
    [
        "training dataset",
        "evaluation dataset",
        "model type",
        "dimensionality",
        "future_disease",
        "mortality",
        "rehospitalization",
    ]
].sort_values(by=["model type", "dimensionality", "training dataset"])

mimic_results.drop(columns=["evaluation dataset"]).to_latex(
    f"{figures_path}/mimic_results_table_{eval_stat}_{eval_model}.tex",
    index=False,
    sparsify=False,
    escape=True,
)
mimic_results.drop(columns=["evaluation dataset"])
# %% ki thrust table

thrust_results = table_decon[
    (table_decon["evaluation dataset"] == "ki_thrust")
    & (
        (table_decon["dataset_variation"] == "mimic_iv_2.2_full")
        | (table_decon["dataset_variation"] == "ki_thrust")
    )
]

remove_encoders = [
    "bert_240_mimic_10",
    "bert_240_mimic_diag",
    "bert_240_mimic_diag_cut8",
    "bert_240_mimic_diag_10",
    "bert_240_mimic_diag_cut7",
]

for encoder in remove_encoders:
    thrust_results = thrust_results[thrust_results["encoder"] != encoder]


thrust_results = thrust_results[
    [
        "training dataset",
        "evaluation dataset",
        "model type",
        # "encoder",
        "dimensionality",
        "future_disease",
        "mortality",
        "rehospitalization",
    ]
].sort_values(by=["model type", "dimensionality", "training dataset"])

thrust_results.drop(columns=["evaluation dataset"]).to_latex(
    f"{figures_path}/ki_thrust_results_table_{eval_stat}_{eval_model}.tex",
    index=False,
    sparsify=False,
    escape=True,
)

thrust_results.drop(columns=["evaluation dataset"])

# %% ablation table


def is_ablation(dataset_variation: str) -> bool:
    return (
        "mimic" in dataset_variation
        and dataset_variation != "mimic_iv_2.2_full"
        and "EHRSHOT" not in dataset_variation
    )


mimic_ablation = table_decon[
    (table_decon["evaluation dataset"] == "mimic")
    & table_decon["dataset_variation"].apply(is_ablation)
]

full_model = table_decon.loc[8, :].to_frame().T
# add full model row to ablation table
mimic_ablation = pd.concat([mimic_ablation, full_model], axis=0, ignore_index=True)

mimic_ablation["ablation"] = mimic_ablation["dataset_variation"].apply(
    lambda x: (
        x.replace("mimic_iv_2.2_", "")
        .replace("mimic_iv_2.2_full", "full model")
        .replace("_full", "")
        .replace("diag_cut", "diagnosis_length")
        .replace("diags", "icd_diagnosis")
        .replace("diag_10", "icd_10_diagnosis")
        .replace("_", " ")
    )
)

# mimic_ablation = pd.concat([mimic_ablation, table_decon.loc[8,:]], axis=0)
mimic_ablation = mimic_ablation[
    [
        # "training dataset",
        # "evaluation dataset",
        # "model type",
        # "encoder",
        "ablation",
        # "dimensionality",
        "future_disease",
        "mortality",
        "rehospitalization",
    ]
].sort_values(by=["ablation"], key=lambda x: x.str.len())

mimic_ablation.to_latex(
    f"{figures_path}/mimic_ablation_table_{eval_stat}_{eval_model}.tex",
    index=False,
    sparsify=False,
    escape=True,
)

mimic_ablation
# %%
mimic_ablation_table = table.xs(key="bert_240", level="encoder")

level = table.index.get_level_values("encoder")
mask = level.str.contains("bert_240_mimic") | (level == "bert_240_ki-thrust")
ki_thrust_ablation_table = table[mask]

wanted = [
    "bert_240_ki-thrust",
    "bert_240_mimic",
    "pat2vec_base",
    "pat2vec_240_ki-thrust",
    "pat2vec_240_mimic",
    "categorical",
    "bert_768_mimic",
    "bert_768_ki-thrust",
]
ki_thrust_table = table[table.index.get_level_values("encoder").isin(wanted)]
ki_thrust_table = ki_thrust_table.xs(key="ki_thrust", level="dataset")

unwanted = [
    "bert_768",
]

mimic_table = table.xs(key="mimic_iv_2.2_full", level="dataset_variation")
# %%

ablation_renames = {
    "bert_240_mimic_diag": "icd diagnosis",
    "bert_240_mimic_diag_10": "icd 10 diagnosis",
    "bert_240_mimic_10": "icd 10",
    "bert_240_mimic_diag_cut8": "diagnosis length 3",
    "bert_240_mimic_diag_cut7": "diagnosis length 2",
    "bert_240_mimic": "full mimic",
    "bert_240_ki-thrust": "full ki-thrust",
}

ki_thrust_ablation_table_print = ki_thrust_ablation_table.copy()
ki_thrust_ablation_table_print.reset_index(inplace=True)
ki_thrust_ablation_table_print["ablation"] = ki_thrust_ablation_table_print[
    "encoder"
].apply(lambda x: ablation_renames.get(x, x))
ki_thrust_ablation_table_print.drop(
    columns=["encoder", "dataset", "dataset_variation"], inplace=True
)
ki_thrust_ablation_table_print = ki_thrust_ablation_table_print[
    ["ablation", "future_disease", "mortality", "rehospitalization"]
].sort_values(by="ablation", key=lambda x: x.str.len())

ki_thrust_ablation_table_print.to_latex(
    f"{figures_path}/ki_thrust_ablation_table_{eval_stat}_{eval_model}.tex",
    index=False,
    sparsify=False,
    escape=True,
)

ki_thrust_ablation_table_print
# %% datasets code types

# icd 10 diag, proc, icd 9 diag, proc, other
# mimic, kithrust, ehrshot

code_type_columns = [
    ("ICD 10", "Diagnosis"),
    ("ICD 10", "Procedure"),
    ("ICD 9", "Diagnosis"),
    ("ICD 9", "Procedure"),
    ("", "Other"),
]

code_type_counts = np.zeros((3, 5), dtype=int)
code_type_counts = pd.DataFrame(
    code_type_counts,
    columns=pd.MultiIndex.from_tuples(code_type_columns),
    index=["MIMIC", "KI-Thrust", "EHRSHOT"],
)
code_type_counts.loc["MIMIC", :] = [1031021, 104466, 1851996, 277120, 0]
code_type_counts.loc["KI-Thrust", :] = [44020813, 0, 0, 0, 0]
code_type_counts.loc["EHRSHOT", :] = [767, 19759, 0, 8468, 41632643]

code_type_precents = code_type_counts.div(code_type_counts.sum(axis=1), axis=0) * 100
code_type_precents
# %%
row_totals = code_type_counts.sum(axis=1)
percentages = code_type_counts.div(row_totals, axis=0) * 100

formatted = (
    code_type_counts.astype(str) + " (" + percentages.round(2).astype(str) + "%)"
)
formatted.T.to_latex(
    f"{figures_path}/code_type_counts.tex",
    index=True,
    header=True,
    multirow=True,
    multicolumn=True,
    sparsify=False,
)
formatted.T
# %%
total_counts = {
    "MIMIC": code_type_counts.loc["MIMIC"].sum(),
    "KI-Thrust": code_type_counts.loc["KI-Thrust"].sum(),
    "EHRSHOT": code_type_counts.loc["EHRSHOT"].sum(),
}

df = code_type_precents.copy()

fig, ax = plt.subplots(figsize=(9, 6))

x_positions = range(len(df))
bottom = [0] * len(df)

for icd_type, category in df.columns:
    values = df[(icd_type, category)]
    ax.bar(df.index, values, bottom=bottom, label=f"{icd_type} - {category}")
    bottom = [b + v for b, v in zip(bottom, values)]

for i, dataset in enumerate(df.index):
    total = total_counts[dataset]
    ax.text(
        i,
        101,
        f"{total:,}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_title("ICD Category Distribution per Dataset")
ax.set_ylabel("Percent of Codes")
ax.set_ylim(0, 110)
ax.legend(title="Code Type & Category", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
