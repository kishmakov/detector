from pathlib import Path
from torch.utils.data import Dataset

from src.db_iterator import db_iterator
from src.embeddings_provider import EmbeddingsProvider


class EmbeddingsDataset(Dataset):
    def __init__(self, embedder: EmbeddingsProvider, database_path: Path, *, source=None, model=None):
        self._embedder = embedder
        self._texts = []  # (text, label)

        for _, _, gold, gens in db_iterator(database_path, source=source, model=model):
            self._texts.append((gold, 1))
            for _, gen_text in gens:
                self._texts.append((gen_text, 0))

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        text, label = self._texts[idx]
        emb = self._embedder.text_to_embeddings(text)
        return emb, label
