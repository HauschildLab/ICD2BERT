# %%
from pathlib import Path

models = [
    "categorical_v2",
    "categorical",
    "CLMBR",
    "beHrt_240",
    "bert_240_ki_thrust",
    "bert_240_mimic_iv_2.2",
    "bert_768_ki_thrust",
    "bert_768_mimic_iv_2.2",
    "med-bert_240",
    "pat2vec_240",
    "pat2vec_240_ki_thrust",
    "pat2vec_base",
]
dataset_variations = [
    variation.name for variation in Path("../../../data/datasets/EHRSHOT").iterdir()
]


array = "\n".join(
    [
        f"python -u src/encode_datasets.py --dataset EHRSHOT --variation {variation} --model {model}"
        for model in models
        for variation in dataset_variations
    ]
)

with open("ehrshot.sh", "w") as f:
    f.write(array)

# %%
