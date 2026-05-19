# %%
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from const import root_dir

data = pd.read_csv(
    root_dir.joinpath(
        "data", "datasets", "mimic_iv_2.2_full", "encoded", "bert_240", "val.csv"
    )
)
data_cat = pd.read_csv(
    root_dir.joinpath(
        "data", "datasets", "mimic_iv_2.2_full", "encoded", "categorical", "val.csv"
    )
)
data.columns
# %%
X_cols = [col for col in data.columns if "X_" in col]
X = data[X_cols]


X_cols_cat = [col for col in data_cat.columns if "X_" in col]
X_cat = data_cat[X_cols_cat]

res = umap.UMAP().fit_transform(StandardScaler().fit_transform(X))
data_t = pd.concat([data, pd.DataFrame(res).add_prefix("UMAP Dim ")], axis=1)

res_cat = umap.UMAP().fit_transform(StandardScaler().fit_transform(X_cat))
data_t_cat = pd.concat(
    [data_cat, pd.DataFrame(res_cat).add_prefix("UMAP Dim ")], axis=1
)
# %%

coloring = "hospital_expire_flag"
coloring = "rehosp_30"

for color, name in [
    ["mortality", "mortality"],
    ["rehospitalization", "rehospitalization"],
]:

    scatter_mort = plt.scatter(
        data_t["UMAP Dim 0"],
        data_t["UMAP Dim 1"],
        c=data_t[color],
        s=10,
        alpha=0.3,
        edgecolor=None,
    )
    plt.title("bert dimensionality reduction " + name)
    plt.legend(*scatter_mort.legend_elements())
    # plt.savefig(f"./data/results/out_tables/bert_dim_red_{name}.pdf", bbox_inches="tight")
    # plt.savefig(f"./data/results/out_tables/bert_dim_red_{name}.png", bbox_inches="tight")
    plt.show()

    scatter = plt.scatter(
        data_t_cat["UMAP Dim 0"],
        data_t_cat["UMAP Dim 1"],
        c=data_t_cat[color],
        s=10,
        alpha=0.3,
        edgecolor=None,
    )
    plt.title("categorical dimensionality reduction " + name)
    plt.legend(*scatter.legend_elements())
    # plt.savefig(f"./data/results/out_tables/cat_dim_red_{name}.pdf", bbox_inches="tight")
    # plt.savefig(f"./data/results/out_tables/cat_dim_red_{name}.png", bbox_inches="tight")
    plt.show()
# %%
from scipy.spatial.distance import pdist
import numpy as np


for data in [data_t, data_t_cat]:
    points = data[data[coloring] == 1][["UMAP Dim 0", "UMAP Dim 1"]]
    points = StandardScaler().fit_transform(points)
    dists = pdist(points)
    print(np.mean(pdist(points)))
    print(np.mean(dists < dists.mean() / 10))
    print()

#    %%
test = StandardScaler().fit_transform(X)
print(test.std(axis=0).shape)
X.head()

# %%
