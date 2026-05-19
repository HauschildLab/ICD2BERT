# %%
from const import root_dir
from multiprocessing import Pool
import gensim
import optuna
from functools import partial
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# %%

train_dataset_path = root_dir.joinpath("data", "datasets", "mimic_iv_2.2", "train.csv")
data = pd.read_csv(train_dataset_path)
model_save_path = root_dir.joinpath("data", "models", "pat2vec_240")
model_save_path.mkdir(parents=True, exist_ok=True)

data_train, data_val = train_test_split(data, test_size=0.1, random_state=0)

best_objective = np.inf


def read_corpus(data: pd.DataFrame, tokens_only=False):
    for i, (_, line) in enumerate(data.iterrows()):
        tokens = line["previous_diagnoses"].split()
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = list(read_corpus(data_train))
val_corpus = list(read_corpus(data_val))


def ranking(model, doc_id):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    return min(rank, 10)


def objective(trial: optuna.trial.Trial):
    global save_path
    global best_objective

    hs = trial.suggest_categorical("hs", [True, False])

    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=240,
        min_count=2,
        max_vocab_size=200_000,
        epochs=trial.suggest_int("epochs", 50, 150),
        window=trial.suggest_int("window", 1, 10),
        negative=trial.suggest_int("negative", 1 - hs, 20),
        ns_exponent=trial.suggest_float("ns_exponent", -5, 5),
        alpha=trial.suggest_float("alpha", 0.001, 0.1, log=True),
        hs=hs,
        dm=trial.suggest_categorical("dm", [True, False]),
        workers=96,
    )
    model.build_vocab(train_corpus, trim_rule=None)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    ranking_model = partial(ranking, model)

    with Pool() as pool:
        ranks = pool.map(ranking_model, range(len(train_corpus) // 10))

    trial_save_path = model_save_path.joinpath(str(trial.params) + ".model")

    model.save(str(trial_save_path))

    objective = sum(ranks) / len(ranks)

    if objective < best_objective:
        best_objective = objective
        model.save(str(model_save_path.joinpath("best.model")))

    return objective


runtime_hours = 24

study = optuna.create_study(study_name="240_vec_mimic")
study.optimize(
    objective,
    timeout=3600 * runtime_hours,
    n_jobs=1,
)
print("----- BEST PARAMETERS -----")
print(study.best_params)
with open(model_save_path.joinpath("best_paramns.txt"), "w") as f:
    f.writelines(str(study.best_params))
