# %%
import pandas as pd
from const import root_dir
import re
from encoder import CategoricalEncoder, CategoricalEncoder_v2


mimic_version = "mimic_iv_2.2"
dataset_path = root_dir.joinpath("data", "datasets", mimic_version)
train_base = pd.read_csv(dataset_path.joinpath("train.csv"))
val_base = pd.read_csv(dataset_path.joinpath("train.csv"))
test_base = pd.read_csv(dataset_path.joinpath("test.csv"))
vocab = pd.read_csv(dataset_path.joinpath("vocab.csv"), header=None, names=["base"])

# bert-for-medical-data/data/models/categorical/icd_categories.csv
# clmbr_translations = pd.read_csv(
#     root_dir.joinpath("data", "EHRSHOT_ASSETS", "icd_to_mimic_code_map.csv")
# )
# clmbr_map = dict(zip(clmbr_translations["mimic_code"], clmbr_translations["code"]))
# vocab["EHRSHOT"] = vocab["base"].map(
#     lambda code: clmbr_map.get(code, clmbr_map.get(code[:-1], "[UNK]"))
# )
cat = CategoricalEncoder(
    root_dir.joinpath("data", "models", "categorical", "icd_categories.csv")
)
cat_v2 = CategoricalEncoder_v2(root_dir.joinpath("data", "models", "categorical"))


def category_map(code, encoder):
    res = encoder.encode([code])
    if res.max(axis=None) == 1:
        return res.idxmax(axis=1).item()
    else:
        return ""


vocab["categories"] = vocab["base"].map(lambda code: category_map(code, cat))
vocab["categories_v2"] = vocab["base"].map(lambda code: category_map(code, cat_v2))


vocab["diags"] = vocab["base"].map(lambda code: code if "d_" in code else "")
vocab["icd_10"] = vocab["base"].map(lambda code: code if "_10_" in code else "")
vocab["diag_10"] = vocab["base"].map(lambda code: code if "d_10_" in code else "")

vocab["diag_cut_3"] = vocab["base"].map(lambda code: code[:8] if "d_" in code else "")
vocab["diag_cut_2"] = vocab["base"].map(lambda code: code[:7] if "d_" in code else "")

vocab_lookup = vocab.copy().set_index(
    "base",
)
vocab_lookup.loc["[SEP]", :] = "[SEP]"

icd_col = "previous_diagnoses"

for vocab_type in vocab_lookup.columns:
    print(vocab_type)

    save_path = dataset_path.parent.joinpath(f"{mimic_version}_{vocab_type}")
    save_path.mkdir(parents=True, exist_ok=True)

    save_path_full = dataset_path.parent.joinpath(f"{mimic_version}_{vocab_type}_full")
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_full.mkdir(parents=True, exist_ok=True)

    vocab_new = vocab[vocab_type].copy()
    vocab_new = vocab_new[vocab_new != ""].drop_duplicates()

    vocab_new.to_csv(save_path.joinpath("vocab.csv"), index=False, header=None)
    vocab_new.to_csv(save_path_full.joinpath("vocab.csv"), index=False, header=None)

    for name, table in [
        ["train.csv", train_base],
        ["val.csv", val_base],
        ["test.csv", test_base],
    ]:
        table_new = table.copy()
        table_new[icd_col] = table[icd_col].map(
            lambda codes: re.sub(
                " +",
                " ",
                " ".join(
                    [vocab_lookup.loc[code, vocab_type] for code in codes.split()]
                ).lstrip(),
            )
        )
        if vocab_type == "EHRSHOT":
            table_new["og_codes"] = table_new[icd_col]
            table_new["patient_id"] = table_new["subject_id"]

        table_new.to_csv(save_path_full.joinpath(name), index=False)
        diag_empty = table_new[icd_col].map(
            lambda codes: set(codes) == set("[SEP] ") or len(codes) <= 5
        )

        table_new[~diag_empty].to_csv(save_path.joinpath(name), index=False)


# %%
