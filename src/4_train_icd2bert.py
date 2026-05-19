# %%
from transformers import (
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BertTokenizer,
)
import sys

import torch
import pandas as pd
from datasets import Dataset

from const import root_dir


import os

print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs visible to PyTorch: {num_gpus}")

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# %%


try:
    hidden_size = int(sys.argv[2])
    print("using hidden size", hidden_size)
except ValueError as e:
    print("ARGV DOES NOT CONTAIN NUMBER USING 240 FOR HIDDEN SIZE INSTEAD")
    hidden_size = 240


if len(sys.argv) >= 2:
    dataset_variation = sys.argv[1]
    print(f"using dataset variation {dataset_variation}")
else:
    dataset_variation = "mimic_iv_2.2"
    print("USING DEFAULT DATASET", dataset_variation)

model_save_path = root_dir.joinpath(
    "data", "models", f"bert_{hidden_size}_{dataset_variation}"
)
vocab_path = root_dir.joinpath("data", "datasets", dataset_variation, "vocab.csv")
train_dataset_path = root_dir.joinpath(
    "data", "datasets", dataset_variation, "train.csv"
)

data = pd.read_csv(train_dataset_path)
dataset_base = Dataset.from_pandas(data[["previous_diagnoses"]])

tokenizer = BertTokenizer(
    vocab_file=vocab_path, do_basic_tokenize=False, truncation_side="left"
)


dataset = dataset_base.map(
    lambda x: tokenizer(
        x["previous_diagnoses"],
        truncation=True,
        max_length=512,
    ),
    batched=True,
)
# %%

dataset = dataset.remove_columns(dataset_base.column_names)

config = BertConfig.from_pretrained("bert-base-uncased")
config.vocab_size = tokenizer.vocab_size + 5
config.hidden_size = hidden_size
config.pad_token_id = tokenizer.pad_token_id
config.sep_token_id = tokenizer.sep_token_id
config.cls_token_id = tokenizer.cls_token_id
config.mask_token_id = tokenizer.mask_token_id
config.unk_token_id = tokenizer.unk_token_id
config.save_pretrained(model_save_path.joinpath("bertconfig"))
model = BertForMaskedLM(config=config)

# %%
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

training_args = TrainingArguments(
    output_dir=model_save_path,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=48,
    save_steps=0.1,
    save_total_limit=10,
    logging_steps=0.01,
    disable_tqdm=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

tokenizer.save_pretrained(model_save_path.joinpath("tokenizer"))
trainer.train()

model.save_pretrained(model_save_path.joinpath("final"))


# %%
