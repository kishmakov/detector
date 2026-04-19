from pathlib import Path
from torch.utils.data import Dataset

from src.json_iterator import json_iterator
from src.model import Model, model_iterator


class EmbeddingsDataset(Dataset):
    def __init__(self, model: Model, data_dir: Path):
        self._model = model
        self._texts = []  # (text, label)

        for _, _, gold, gen in json_iterator(data_dir):
            if isinstance(gen, str):
                self._texts.append((gold, 1))
                self._texts.append((gen, 0))

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        text, label = self._texts[idx]
        emb = self._model.text_to_embeddings(text)
        return emb, label


def dataset_iterator(data_dir: Path):
    for model in model_iterator():
        yield model.short_id, EmbeddingsDataset(model, data_dir)
