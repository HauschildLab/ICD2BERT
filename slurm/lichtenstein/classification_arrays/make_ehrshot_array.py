# %%
from pathlib import Path

encoders = [
    "CLMBR",
    "beHrt_240",
    "bert_240_ki_thrust",
    "bert_240_mimic_iv_2.2",
    "bert_768_ki_thrust",
    "bert_768_mimic_iv_2.2",
    "categorical",
    "categorical_v2",
    "med-bert_240",
    "pat2vec_240",
    "pat2vec_240_ki_thrust",
    "pat2vec_base",
]
tasks = [p.name for p in Path("../../../data/datasets/EHRSHOT").iterdir()]
tasks.remove("chexpert")
shuffle_seeds = [0, 1, 2, 3, 4]
array = "\n".join(
    [
        f"python -u src/classification_ehr_shot.py --task {task} --encoder {encoder} --shuffle_seed {shuffle_seed}"
        for task in tasks
        for encoder in encoders
        for shuffle_seed in shuffle_seeds
    ]
)
array_chex = "\n".join(
    [
        f"python -u src/classification_ehr_shot.py --task chexpert --encoder {encoder} --shuffle_seed {shuffle_seed} --chexpert_label {lab}"
        for lab in range(14)
        for encoder in encoders
        for shuffle_seed in shuffle_seeds
    ]
)


with open("ehrshot.sh", "w") as f:
    f.write(array + "\n" + array_chex)

# %%
