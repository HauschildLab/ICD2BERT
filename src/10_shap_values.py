# %%
from sklearn.ensemble import RandomForestClassifier
from encoder import BertEncoder
import pickle
import pandas as pd
import numpy as np
import shap
import torch
import weasyprint
from bertviz import head_view
from IPython.display import display, HTML as IHTML
from const import root_dir
import matplotlib.pyplot as plt

save_path = root_dir.joinpath("data", "results", "shap")
figures_path = root_dir.joinpath("data", "results", "tables_and_figures")

# ── Load encoder and data ─────────────────────────────────────────────────────
encoder = BertEncoder(
    root_dir.joinpath("data", "models", "bert_240_mimic_iv_2.2", "checkpoint-115800"),
)

with open(
    root_dir.joinpath(
        "data",
        "results",
        "classification_stats",
        "mimic",
        "mortality",
        "mimic_iv_2.2_full",
        "bert_240",
        "classifiers",
        "RandomForest",
        "resample_seed_0",
        "best_model.pkl",
    ),
    "rb",
) as f:
    clf_mortality = pickle.load(f)
    assert isinstance(clf_mortality, RandomForestClassifier)

with open(
    root_dir.joinpath(
        "data",
        "results",
        "classification_stats",
        "mimic",
        "rehospitalization",
        "mimic_iv_2.2_full",
        "bert_240",
        "classifiers",
        "RandomForest",
        "resample_seed_0",
        "best_model.pkl",
    ),
    "rb",
) as f:
    clf_rehospitalization = pickle.load(f)
    assert isinstance(clf_rehospitalization, RandomForestClassifier)

data = pd.read_csv(
    root_dir.joinpath("data", "datasets", "mimic_iv_2.2_full", "val.csv")
)
text_col = "previous_diagnoses"

# %%
# Encode all validation samples once to find representative examples
embeddings = encoder.encode(data[text_col].tolist(), batch_size=32)
embeddings = embeddings.rename(columns=lambda x: f"X_bert_240_mimic_iv_2.2_{x}")

# %%
# Find indices of correctly predicted True/False samples for each task
mortality_preds = clf_mortality.predict(embeddings)
rehospitalization_preds = clf_rehospitalization.predict(embeddings)


def count_icd_codes(text):
    return sum(1 for t in str(text).split() if t != "[SEP]")


def first_correct(preds, labels, target, max_codes=30, min_codes=15):
    mask = (preds == target) & (labels.values == target)
    for idx in np.where(mask)[0]:
        if min_codes <= count_icd_codes(data[text_col].iloc[idx]) <= max_codes:
            return int(idx)
    raise ValueError(
        f"No correctly predicted sample with target={target} and <= {max_codes} ICD codes found."
    )


mortality_true_idx = first_correct(mortality_preds, data["mortality"], 1)
mortality_false_idx = first_correct(mortality_preds, data["mortality"], 0)
rehospitalization_true_idx = first_correct(
    rehospitalization_preds, data["rehospitalization"], 1
)
rehospitalization_false_idx = first_correct(
    rehospitalization_preds, data["rehospitalization"], 0
)

print(
    f"mortality      - True idx: {mortality_true_idx}, False idx: {mortality_false_idx}"
)
print(
    f"rehospitalization - True idx: {rehospitalization_true_idx}, False idx: {rehospitalization_false_idx}"
)

# %%
# predict_proba factory - returns P(False) and P(True) for a given classifier
EMBED_PREFIX = "X_bert_240_mimic_iv_2.2_"


def make_predict_proba(clf, clf_cols):
    def predict_proba(texts):
        embs = encoder.encode(list(texts), batch_size=32)
        embs = embs.rename(columns=lambda x: f"{EMBED_PREFIX}{x}")
        probas = clf.predict_proba(embs)  # shape (n, 2): [P(False), P(True)]
        return pd.DataFrame(probas, columns=clf_cols)

    return predict_proba


mortality_cols = ["mortality=False", "mortality=True"]
rehospitalization_cols = ["rehospitalization=False", "rehospitalization=True"]

predict_proba_mortality = make_predict_proba(clf_mortality, mortality_cols)
predict_proba_rehospitalization = make_predict_proba(
    clf_rehospitalization, rehospitalization_cols
)

# %%
# Build SHAP explainers
explainer_mortality = shap.Explainer(
    predict_proba_mortality,
    encoder.tokenizer,
    batch_size=32,
    output_names=mortality_cols,
)
explainer_rehospitalization = shap.Explainer(
    predict_proba_rehospitalization,
    encoder.tokenizer,
    batch_size=32,
    output_names=rehospitalization_cols,
)

# %%
# Compute SHAP values for the 4 representative samples
shap_mortality_true = explainer_mortality([data[text_col].iloc[mortality_true_idx]])
shap_mortality_false = explainer_mortality([data[text_col].iloc[mortality_false_idx]])
shap_rehospitalization_true = explainer_rehospitalization(
    [data[text_col].iloc[rehospitalization_true_idx]]
)
shap_rehospitalization_false = explainer_rehospitalization(
    [data[text_col].iloc[rehospitalization_false_idx]]
)

# %%
# 4 SHAP text plots
shap.plots.text(shap_mortality_true[0])
shap.plots.text(shap_mortality_false[0])
shap.plots.text(shap_rehospitalization_true[0])
shap.plots.text(shap_rehospitalization_false[0])

# %%
# Save all 4 SHAP value objects
save_path.mkdir(parents=True, exist_ok=True)

shap_bundle = {
    "mortality_true": shap_mortality_true,
    "mortality_false": shap_mortality_false,
    "rehospitalization_true": shap_rehospitalization_true,
    "rehospitalization_false": shap_rehospitalization_false,
}
with open(save_path.joinpath("forest_mimic.pkl"), "wb") as f:
    pickle.dump(shap_bundle, f)

# %%
# Load SHAP values
with open(save_path.joinpath("forest_mimic.pkl"), "rb") as f:
    shap_bundle = pickle.load(f)

shap_mortality_true = shap_bundle["mortality_true"]
shap_mortality_false = shap_bundle["mortality_false"]
shap_rehospitalization_true = shap_bundle["rehospitalization_true"]
shap_rehospitalization_false = shap_bundle["rehospitalization_false"]

# %%
# Export all 4 plots to PDF and combined HTML
figures_path.mkdir(parents=True, exist_ok=True)

labels = {
    "mortality_true": "Mortality - correctly predicted True",
    "mortality_false": "Mortality - correctly predicted False",
    "rehospitalization_true": "Rehospitalization - correctly predicted True",
    "rehospitalization_false": "Rehospitalization - correctly predicted False",
}

plot_htmls = []
for name, sv in shap_bundle.items():
    html = shap.plots.text(sv[0], display=False)
    weasyprint.HTML(string=html).write_pdf(
        str(figures_path / f"shap_text_mimic_{name}.pdf")
    )
    plot_htmls.append(f"<h2 style='font-family:sans-serif'>{labels[name]}</h2>\n{html}")

combined_html = "<html><body>\n" + "\n<hr>\n".join(plot_htmls) + "\n</body></html>"
with open(figures_path / "shap_text_mimic_all.html", "w", encoding="utf-8") as f:
    f.write(combined_html)

# %%
# Beeswarm plots - top 10 ICD codes by mean |SHAP| across N_BEESWARM_SAMPLES samples

N_BEESWARM_SAMPLES = 100
rng = np.random.default_rng(42)
beeswarm_idx = rng.choice(len(data), size=N_BEESWARM_SAMPLES, replace=False)
beeswarm_texts = data[text_col].iloc[beeswarm_idx].tolist()


def _is_special_token(t):
    s = t.strip()
    return not s or s.startswith("[")


def shap_to_token_matrix(explainer, texts, output_idx=1):
    """
    Returns (shap_df, presence_df): rows = samples, columns = ICD-code tokens.
    shap_df holds the SHAP value for each token in each sample (0 if absent).
    presence_df holds 1 where the token appears, 0 otherwise (used for coloring).
    output_idx=1 selects the =True class.
    """
    sv = explainer(texts)
    shap_rows, presence_rows = [], []
    for i in range(len(sv)):
        tokens = list(sv[i].data)
        values = sv[i].values[:, output_idx]
        shap_row, presence_row = {}, {}
        for t, v in zip(tokens, values):
            if _is_special_token(t):
                continue
            shap_row[t] = shap_row.get(t, 0) + v
            presence_row[t] = 1
        shap_rows.append(shap_row)
        presence_rows.append(presence_row)
    return pd.DataFrame(shap_rows).fillna(0), pd.DataFrame(presence_rows).fillna(0)


def make_beeswarm_explanation(shap_df, presence_df, k=10, min_presence=10):
    frequent = presence_df.columns[presence_df.sum() >= min_presence]
    top_k = (
        shap_df[frequent]
        .abs()
        .where(presence_df[frequent] == 1)
        .mean()
        .nlargest(k)
        .index.tolist()
    )
    return shap.Explanation(
        values=shap_df[top_k].values,
        data=presence_df[top_k].values,
        feature_names=top_k,
    )


shap_df_mortality, presence_df_mortality = shap_to_token_matrix(
    explainer_mortality, beeswarm_texts
)
shap_df_rehospitalization, presence_df_rehospitalization = shap_to_token_matrix(
    explainer_rehospitalization, beeswarm_texts
)


# %%
plt.close("all")
shap.plots.beeswarm(
    make_beeswarm_explanation(shap_df_mortality, presence_df_mortality), show=False
)
plt.title("10 highest impact ICD codes for mortality prediction")
plt.show()
# %%
plt.close("all")
shap.plots.beeswarm(
    make_beeswarm_explanation(shap_df_rehospitalization, presence_df_rehospitalization),
    show=False,
)
plt.title("10 highest impact ICD codes for rehospitalization prediction")
plt.show()

# %%
# Save beeswarm data
beeswarm_bundle = {
    "mortality": {"shap": shap_df_mortality, "presence": presence_df_mortality},
    "rehospitalization": {
        "shap": shap_df_rehospitalization,
        "presence": presence_df_rehospitalization,
    },
}
with open(save_path.joinpath(f"beeswarm_mimic_{N_BEESWARM_SAMPLES}.pkl"), "wb") as f:
    pickle.dump(beeswarm_bundle, f)

# %%
# Load beeswarm data
with open(save_path.joinpath(f"beeswarm_mimic_{N_BEESWARM_SAMPLES}.pkl"), "rb") as f:
    beeswarm_bundle = pickle.load(f)

shap_df_mortality = beeswarm_bundle["mortality"]["shap"]
presence_df_mortality = beeswarm_bundle["mortality"]["presence"]
shap_df_rehospitalization = beeswarm_bundle["rehospitalization"]["shap"]
presence_df_rehospitalization = beeswarm_bundle["rehospitalization"]["presence"]

# %%
# Export beeswarm plots to PDF
for task_name, dfs in beeswarm_bundle.items():
    plt.close("all")
    explanation = make_beeswarm_explanation(
        dfs["shap"], dfs["presence"], min_presence=5
    )
    shap.plots.beeswarm(explanation, show=False)
    plt.title(f"{task_name.capitalize()} - Top 10 ICD Codes")
    plt.tight_layout()
    plt.savefig(
        str(figures_path / f"beeswarm_mimic_{task_name}_{N_BEESWARM_SAMPLES}.pdf")
    )
    plt.close()

# %%
# Attention visualization on the mortality=True sample
sample_text = data[text_col].iloc[mortality_true_idx]
filtered_text = " ".join(t for t in str(sample_text).split() if t in encoder.vocab)

inputs = encoder.tokenizer(
    filtered_text, return_tensors="pt", truncation=True, max_length=512
).to(encoder.device)

with torch.no_grad():
    outputs = encoder.bert(**inputs, output_attentions=True)

tokens = encoder.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

TOKEN_BOX_WIDTH = 120

html = head_view(outputs.attentions, tokens, html_action="return")
html_wide = html.data.replace(
    "const BOXWIDTH = 110", f"const BOXWIDTH = {TOKEN_BOX_WIDTH}"
)

display(IHTML(html_wide))

# %%
weasyprint.HTML(string=html_wide).write_pdf(
    str(figures_path / "attention_patterns.pdf")
)
