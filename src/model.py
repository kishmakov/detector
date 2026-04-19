import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_INFO = [
    ("roberta-base",               {"style": "-",                   "short_id": "robb"}),
    ("roberta-large",              {"style": "--",                  "short_id": "robl"}),
    ("bert-base-uncased",          {"style": "-.",                  "short_id": "berb"}),
    ("bert-large-uncased",         {"style": ":",                   "short_id": "berl"}),
    ("distilroberta-base",         {"style": (0, (5, 1)),           "short_id": "disbn"}),
    ("distilbert-base-uncased",    {"style": (0, (3, 1, 1, 1)),     "short_id": "disbu"}),
    ("microsoft/deberta-v3-base",  {"style": (0, (7, 2)),           "short_id": "debb"}),
    ("microsoft/deberta-v3-large", {"style": (0, (3, 2, 1, 2)),     "short_id": "debl"}),
]

CHUNK_SIZE = 510

class Model:
    def __init__(self, name: str, info: dict):
        self.name = name
        self.short_id = info["short_id"]
        self.style = info["style"]
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModel.from_pretrained(name)
        self._model.eval()

    def text_to_embeddings(self, text: str, max_length: int = 512) -> np.ndarray:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)[:max_length]
        token_ids = token_ids

        chunks = []
        for i in range(0, len(token_ids), CHUNK_SIZE):
            chunk = token_ids[i:i + CHUNK_SIZE]
            ids = torch.tensor([[self._tokenizer.cls_token_id] + chunk + [self._tokenizer.sep_token_id]])
            with torch.no_grad():
                out = self._model(input_ids=ids, attention_mask=torch.ones_like(ids))
            chunks.append(out[0][0, 1:-1].cpu().numpy())
        return np.concatenate(chunks, axis=0)


def model_iterator():
    for name, info in MODEL_INFO:
        yield Model(name, info)
