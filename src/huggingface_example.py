# %%
from transformers import BertForMaskedLM, BertTokenizer

model = BertForMaskedLM.from_pretrained(
    "JonasHri/ICD2BERT",
    subfolder="model",
)

tokenizer = BertTokenizer.from_pretrained(
    "JonasHri/ICD2BERT",
    subfolder="tokenizer",
)


def encode(icd_sequence: str, model: BertForMaskedLM, tokenizer: BertTokenizer):
    inputs = tokenizer(
        icd_sequence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    outputs = model(**inputs, output_hidden_states=True)

    embedding = outputs.hidden_states[-1][:, 0, :]
    return embedding


icd_sequence = (
    [
        "d_10_T1404 d_10_I82532 d_10_Z7901 d_10_Z86711 d_10_S0628 d_10_I872 d_10_F329 d_10_E669 d_10_Z6834 d_10_Z590 ",
        "p_09_5122 d_09_57400 d_09_V6441 d_09_78829 d_09_53081 ",
    ],
)

embedding = encode(icd_sequence, model=model, tokenizer=tokenizer)
print(embedding[0])
print(embedding[1])
