# %%
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
    "bert_240_mimic_iv_2.2_diag_10",
    "bert_240_mimic_iv_2.2_categoriesbert_240_mimic_iv_2.2_categories_v2",
    "bert_240_mimic_iv_2.2_diag_10",
    "bert_240_mimic_iv_2.2_diag_cut_2",
    "bert_240_mimic_iv_2.2_diag_cut_3",
    "bert_240_mimic_iv_2.2_diags",
    "bert_240_mimic_iv_2.2_icd_10",
    "bert_base_mimic_sep_240",
]
dataset_variations = ["normal"]


array = "\n".join(
    [
        f"python -u src/encode_datasets.py --dataset mimic_iv_2.2 --variation {variation} --model {model}"
        for model in models
        for variation in dataset_variations
    ]
)

with open("mimic.sh", "w") as f:
    f.write(array)

# %%
