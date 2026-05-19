# %%
from const import root_dir
import torch
from transformers import BertForMaskedLM, BertTokenizer
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
from behrt import Behrt
import femr.models.transformer
import femr.models.tokenizer
import femr.models.processor
import datetime


class CLMBrEncoder:
    def __init__(
        self,
        model_name: str = "StanfordShahLab/clmbr-t-base",
        device="cuda",
    ) -> None:

        self.devie = device
        # Load tokenizer / batch loader
        self.tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(model_name)
        self.batch_processor = femr.models.processor.FEMRBatchProcessor(self.tokenizer)

        # Load model
        self.model = femr.models.transformer.FEMRModel.from_pretrained(model_name).to(
            device
        )

    def encode(self, dataframe: pd.DataFrame, silent: bool = True):

        data = dataframe[["patient_id", "og_codes", "timestamps", "birth_DATETIME"]]

        representations = []
        for row in tqdm(data.itertuples(), total=len(data), disable=silent):

            birth_time = [datetime.datetime.strptime(row.birth_DATETIME, "%Y-%m-%d")]
            birth_code = ["SNOMED/184099003"]

            codes = birth_code + row.og_codes.split(" ")

            time_snippets = row.timestamps.strip().split(" ")
            timestrings = [
                " ".join(time_snippets[i : i + 2])
                for i in range(0, len(time_snippets), 2)
            ]
            try:
                timestamps = birth_time + list(
                    map(
                        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
                        timestrings,
                    )
                )
            except ValueError as e:
                print(f"Error parsing timestamps for patient {row.patient_id}: {e}")
                print(f"Timestamps: {timestrings}")
                raise e
            patient = {
                "patient_id": row.patient_id,
                "events": [
                    {
                        "time": timestamps[i],
                        "measurements": [{"code": codes[i]}],
                    }
                    for i in range(len(codes))
                ],
            }

            raw_batch = self.batch_processor.convert_patient(patient, tensor_type="pt")
            batch = self.batch_processor.collate([raw_batch])

            for key, value in batch["batch"].items():
                if isinstance(value, torch.Tensor):
                    batch["batch"][key] = value.to(self.devie)

            for key, value in batch["batch"]["transformer"].items():
                if isinstance(value, torch.Tensor):
                    batch["batch"]["transformer"][key] = value.to(self.devie)

            with torch.no_grad():
                _, result = self.model(**batch)
                representations.append(result["representations"][-1].cpu().numpy())

        result_df = pd.DataFrame(representations)
        return result_df


class BertEncoder:
    def __init__(
        self,
        bert_path: Path,
        use_cls: bool = True,
        device=None,
    ) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert = BertForMaskedLM.from_pretrained(bert_path).to(self.device)
        self.bert.eval()
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_path.parent.joinpath("tokenizer"), use_fast=False
        )
        self.use_cls = use_cls
        self.vocab = set(self.tokenizer.vocab)
        self.vocab.add("[SEP]")

    def encode(self, uid_str: List[str], batch_size=16, silent: bool = True):
        batch_position = 0

        results = []

        with tqdm(total=len(uid_str), disable=silent) as pbar:
            while batch_position < len(uid_str):
                pbar.update(batch_size)
                batch = uid_str[batch_position : batch_position + batch_size]
                batch = [
                    " ".join(token for token in text.split() if token in self.vocab)
                    for text in batch
                ]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.bert(**inputs, output_hidden_states=True)

                if self.use_cls:
                    batch_embeding = outputs.hidden_states[-1][:, 0, :].cpu()
                else:
                    batch_embeding = torch.sum(outputs.hidden_states[-1], dim=1).cpu()
                    lengths = list(map(lambda x: max(len(x.split()), 1), batch))
                    batch_embeding = batch_embeding / torch.Tensor(lengths)[:, None]
                results.append(batch_embeding)
                batch_position += batch_size
        return pd.DataFrame(torch.vstack(results))


class BehrtEncoder:
    def __init__(
        self,
        bert_path: str = "./data/models/beHrt_base_ki-thrust_240/checkpoint-102352",
        use_cls: bool = True,
        device=None,
        verbose=False,
    ) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.bert = Behrt.from_pretrained(bert_path).to(self.device)
        self.bert.load_embeddings(str(bert_path), verbose=verbose, device=self.device)
        self.bert.eval()
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_path.parent.joinpath("tokenizer")
        )
        self.use_cls = use_cls
        self.vocab = {x for x in self.tokenizer.vocab}
        self.vocab.add("[SEP]")
        self.verbose = verbose

    def encode(
        self,
        uid_str: List[str],
        age_ids: List[str],
        pos_ids: List[str],
        seg_ids: List[str],
        seg_alt_ids: List[str],
        batch_size=16,
    ):
        batch_point = 0

        results = []

        while batch_point < len(uid_str):
            if self.verbose:
                print(batch_point / len(uid_str) * 100, "%")
            batch_uid = uid_str[batch_point : batch_point + batch_size]
            batch_age_ids = age_ids[batch_point : batch_point + batch_size]
            batch_pos_ids = pos_ids[batch_point : batch_point + batch_size]
            batch_seg_ids = seg_ids[batch_point : batch_point + batch_size]
            batch_seg_alt_ids = seg_alt_ids[batch_point : batch_point + batch_size]

            batch = []
            batch_age = []
            batch_pos = []
            batch_seg = []
            batch_seg_alt = []
            for uids, age, pos, seg, seg_alt in zip(
                batch_uid,
                batch_age_ids,
                batch_pos_ids,
                batch_seg_ids,
                batch_seg_alt_ids,
            ):
                token_filter = [token in self.vocab for token in uids.split()]
                batch.append(
                    " ".join(
                        token
                        for token, filtr in zip(uids.split(), token_filter)
                        if filtr
                    )
                )

                # print(batch[-1])
                # print(age)
                # print(token_filter)

                # filter and truncation
                batch_age.append(
                    [
                        int(token)
                        for token, filtr in zip(age.split(), token_filter)
                        if filtr
                    ][-512:]
                )
                batch_pos.append(
                    [
                        int(token)
                        for token, filtr in zip(pos.split(), token_filter)
                        if filtr
                    ][:512]
                )
                batch_seg.append(
                    [
                        int(token)
                        for token, filtr in zip(seg.split(), token_filter)
                        if filtr
                    ][-512:]
                )
                batch_seg_alt.append(
                    [
                        int(token)
                        for token, filtr in zip(seg_alt.split(), token_filter)
                        if filtr
                    ][-512:]
                )

                # padding
                batch_age[-1] = batch_age[-1] + [0] * (512 - len(batch_age[-1]))
                batch_pos[-1] = batch_pos[-1] + [0] * (512 - len(batch_pos[-1]))
                batch_pos[-1] = list(range(1, 513))
                batch_seg[-1] = batch_seg[-1] + [0] * (512 - len(batch_seg[-1]))
                batch_seg_alt[-1] = batch_seg_alt[-1] + [2] * (
                    512 - len(batch_seg_alt[-1])
                )

            # print(batch)
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                # padding="max_length",
                max_length=512,
            )

            # print(self.tokenizer.batch_decode(inputs["input_ids"])[-1])

            inputs["age_ids"] = torch.tensor(batch_age, dtype=torch.int32)
            inputs["pos_ids"] = torch.tensor(batch_pos, dtype=torch.int32)
            inputs["seg_ids"] = torch.tensor(batch_seg, dtype=torch.int32)
            inputs["seg_ids_alt"] = torch.tensor(batch_seg_alt, dtype=torch.int32)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.bert(**inputs, output_hidden_states=True)

            if self.use_cls:
                batch_embeding = outputs.hidden_states[-1][:, 0, :].cpu()
            else:
                lengths = list(map(lambda x: len(x.split()) + 2, batch))

                batch_embeddings = outputs.hidden_states[-1]

                batch_embeding = torch.stack(
                    [
                        batch_embeddings[i, : lengths[i]].mean(dim=0)
                        for i in range(len(lengths))
                    ]
                )
                # print(batch)
                # print(outputs.hidden_states[-1].shape)
                # batch_embeding = torch.sum(outputs.hidden_states[-1], dim=1).cpu()
                # print(outputs.hidden_states[-1][2,6,:])
                # print(batch_embeding.shape)
                # print(lengths)
                # batch_embeding = batch_embeding / torch.Tensor(lengths)[:, None]

            results.append(batch_embeding.cpu())
            batch_point += batch_size
        return pd.DataFrame(torch.vstack(results))


class Pat2VecEncoder:
    def __init__(self, path="", only_d_10=False) -> None:
        self.pat2vec_model = Doc2Vec.load(path)
        self.only_d_10 = only_d_10

    def encode(self, uid_str: List[str]):
        if type(uid_str) == str:
            uid_str = [uid_str]

        icds = []

        for val in uid_str:
            codes = val.split(" ")
            if self.only_d_10:
                diag_10 = filter(lambda x: "d_10_" in x, codes)
                diag_10 = map(lambda x: x.replace("d_10_", ""), diag_10)
            else:
                diag_10 = codes
            icds.append(diag_10)

        embeddings = torch.Tensor(
            np.array([self.pat2vec_model.infer_vector(val) for val in icds])
        )
        return pd.DataFrame(embeddings)


class CategoricalEncoder_v2:
    def __init__(self, categories_path: Path = None) -> None:
        datapath = (
            categories_path
            if categories_path is not None
            else root_dir.joinpath(
                "data",
                "models",
                "categorical",
            )
        )
        code_category_to_file = {
            "d_10": "icd_categories.csv",
            "d_09": "diag_9_categories.csv",
            "p_10": "proc_10_categories.csv",
            "p_09": "proc_9_categories.csv",
        }

        self.translator = {}
        self.categories = []
        for code_type, filename in code_category_to_file.items():
            data = pd.read_csv(datapath.joinpath(filename))
            for _, row in data.iterrows():
                category = f"{code_type}_{row['start']}-{row['end']}"
                self.categories.append(category)
                if "d_" in code_type:
                    letter = row["start"][0]
                    start_num = int(row["start"][1:])
                    end_num = int(row["end"][1:])

                    if letter == row["end"][0]:
                        for i in range(start_num, end_num + 1):
                            self.translator[f"{code_type}_{letter}{i:02d}"] = category
                    else:
                        for i in range(start_num, 100):
                            self.translator[f"{code_type}_{letter}{i:02d}"] = category
                        for i in range(0, end_num + 1):
                            self.translator[f"{code_type}_{row['end'][0]}{i:02d}"] = (
                                category
                            )
                elif "p_" in code_type:
                    self.translator[f"{code_type}_{row['start']}"] = category

    def to_int(self, code):
        try:
            return int(code[6:8])
        except:
            try:
                return int(code[6])
            except:
                return -1

    def encode(self, text: List[str]) -> pd.DataFrame:
        results = pd.DataFrame(
            np.zeros((len(text), len(self.categories)), dtype=int),
            columns=self.categories,
        )
        for i, sentence in enumerate(text):
            codes = set()
            words = sentence.split()
            for word in words:
                if "p_" in word:
                    word = word[:7]
                elif "d_" in word:
                    word = word[:8]
                codes.add(self.translator.get(word))

            if None in codes:
                codes.remove(None)
            results.loc[i, list(codes)] = 1

        return results


class CategoricalEncoder:
    def __init__(self, categories_path=None) -> None:
        datapath = (
            categories_path
            if categories_path is not None
            else root_dir.joinpath(
                "data", "models", "categorical", "icd_categories.csv"
            )
        )
        self.data = pd.read_csv(datapath)
        self.categories = []
        self.translator = {}
        for _, row in self.data.iterrows():
            category = f"{row['start']}-{row['end']}"
            self.categories.append(category)

            letter = row["start"][0]
            start_num = int(row["start"][1:])
            end_num = int(row["end"][1:])

            for i in range(start_num, end_num + 1):
                self.translator[(letter, i)] = category

    def to_int(self, code):
        try:
            return int(code[6:8])
        except:
            try:
                return int(code[6])
            except:
                return -1

    def encode(self, text: List[str]) -> pd.DataFrame:
        results = pd.DataFrame(
            np.zeros((len(text), len(self.data)), dtype=int),
            columns=self.categories,
        )
        for i, sentence in enumerate(text):
            codes = set()
            words = list(filter(lambda x: x.startswith("d_10_"), sentence.split()))
            letters = list(map(lambda x: x[5], words))
            numbers = list(map(lambda x: self.to_int(x), words))
            for letter, number in zip(letters, numbers):
                codes.add(self.translator.get((letter, number)))

            if None in codes:
                codes.remove(None)
            results.loc[i, list(codes)] = 1

        return results


if __name__ == "__main__":

    # encoder = BertEncoder(root_dir.joinpath("data", "models", "icbert_240", "final"))
    encoder = CLMBrEncoder(device="cuda")

    data = pd.read_csv(
        root_dir.joinpath("data", "datasets", "EHRSHOT", "guo_icu", "val.csv")
    )

    res = encoder.encode(data.head(100), silent=False)

    # res = encoder.encode(
    #     [
    #         "d_10_T1404 d_10_I82532 d_10_Z7901 d_10_Z86711 d_10_S0628 d_10_I872 d_10_F329 d_10_E669 d_10_Z6834 d_10_Z590 ",
    #         "p_09_5122 d_09_57400 d_09_V6441 d_09_78829 d_09_53081 ",
    #         "",
    #     ],
    # )
    print(res)

# %%
