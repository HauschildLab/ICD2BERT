from torch import nn
from transformers import BertForMaskedLM, BertConfig
from safetensors import safe_open
import torch


class Behrt(BertForMaskedLM):
    def __init__(
        self,
        config: BertConfig,
        num_age_embeddings=None,
        num_pos_embeddings=None,
        num_seg_embeddings=None,
        num_seg_alt_embeddings=None,
        default_way=False,
    ):
        super().__init__(config)

        self.default_way = default_way

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.age_embeddings = (
            nn.Embedding(num_age_embeddings, config.hidden_size, padding_idx=0)
            if num_age_embeddings
            else None
        )

        self.pos_embeddings = (
            nn.Embedding(num_pos_embeddings, config.hidden_size, padding_idx=0)
            if num_pos_embeddings
            else None
        )

        self.seg_embeddings = (
            nn.Embedding(num_seg_embeddings, config.hidden_size, padding_idx=0)
            if num_seg_embeddings
            else None
        )

        self.seg_alt_embeddings = (
            nn.Embedding(num_seg_alt_embeddings, config.hidden_size, padding_idx=2)
            if num_seg_alt_embeddings
            else None
        )

    def copy_pretrain_weights(self):
        with torch.no_grad():
            self.word_embeddings.weight.copy_(
                self.bert.embeddings.word_embeddings.weight
            )
        # self.word_embeddings.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        age_ids=None,
        pos_ids=None,
        seg_ids=None,
        seg_ids_alt=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):

        if self.default_way:
            # print("using default way")
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        # print(type(input_ids), type(age_ids))
        # print(input_ids.shape, age_ids.shape)
        input_length = input_ids.shape[1]

        full_embs = self.word_embeddings(input_ids)

        if self.age_embeddings:
            age_ids = age_ids[:, :input_length]
            age_embs = self.age_embeddings(age_ids)
            full_embs = full_embs + age_embs

        if self.pos_embeddings:
            pos_ids = pos_ids[:, :input_length]
            pos_embs = self.pos_embeddings(pos_ids)
            full_embs = full_embs + pos_embs

        if self.seg_embeddings:
            seg_ids = seg_ids[:, :input_length]
            seg_embs = self.seg_embeddings(seg_ids)
            full_embs = full_embs + seg_embs

        if self.seg_alt_embeddings:
            seg_ids_alt = seg_ids_alt[:, :input_length]
            seg_alt_embs = self.seg_alt_embeddings(seg_ids_alt)
            full_embs = full_embs + seg_alt_embs

        # full_embs = self.bert.embeddings.LayerNorm(full_embs)
        # full_embs = self.bert.embeddings.dropout(full_embs)

        return super().forward(
            inputs_embeds=full_embs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

    def load_embeddings(self, bert_path: str, device="cuda", verbose=False):

        def make_embedding(tensor, verbose=False):

            count, dims = tensor.shape
            if verbose:
                print("word", count, dims)

            embedding = nn.Embedding(count, dims)
            embedding.weight = nn.Parameter(tensor)
            return embedding

        with safe_open(
            bert_path + "/model.safetensors", framework="pt", device="cpu"
        ) as f:
            for k in f.keys():
                if k == "word_embeddings.weight":
                    tensor = f.get_tensor(k)
                    self.word_embeddings = make_embedding(tensor, verbose=verbose).to(
                        device
                    )

                if k == "age_embeddings.weight":
                    tensor = f.get_tensor(k)
                    self.age_embeddings = make_embedding(tensor, verbose=verbose).to(
                        device
                    )

                if k == "pos_embeddings.weight":
                    tensor = f.get_tensor(k)
                    self.pos_embeddings = make_embedding(tensor, verbose=verbose).to(
                        device
                    )

                if k == "seg_embeddings.weight":
                    tensor = f.get_tensor(k)
                    self.seg_embeddings = make_embedding(tensor, verbose=verbose).to(
                        device
                    )

                if k == "seg_alt_embeddings.weight":
                    tensor = f.get_tensor(k)
                    self.seg_alt_embeddings = make_embedding(
                        tensor, verbose=verbose
                    ).to(device)
                    # count, dims = tensor.shape
                    # if verbose:
                    #     print("seg alt", count, dims)

                    # self.seg_alt_embeddings = nn.Embedding(count, dims)
                    # self.seg_alt_embeddings.weight = nn.Parameter(tensor)
                    # self.seg_alt_embeddings = self.seg_alt_embeddings.to(device)
