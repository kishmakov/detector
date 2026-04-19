import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.json_iterator import json_iterator
from src.model import model_iterator

DATA_DIR = Path(__file__).parent.parent / "main_paper_data" / "data"
OUT_DIR = Path(__file__).parent.parent / "data" / "embeddings"
MAX_LENGTH = 512

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-group entries by file to avoid re-iterating per model
by_file = {}
for file_name, prefix, gold_completion, gen_completion in json_iterator(DATA_DIR):
    if isinstance(gen_completion, str):
        by_file.setdefault(file_name, []).append((gold_completion, gen_completion))

for model in model_iterator():
    print(f"Loading {model.name} ...")

    for file_name, entries in by_file.items():
        n = len(entries)
        print(f"  {file_name} has {n} entries")

        gold_embeddings = np.empty(n, dtype=object)
        gen_embeddings = np.empty(n, dtype=object)

        for i, (gold, gen) in enumerate(entries):
            gold_embeddings[i] = model.text_to_embeddings(gold, MAX_LENGTH)
            gen_embeddings[i] = model.text_to_embeddings(gen, MAX_LENGTH)

        np.save(OUT_DIR / f"{file_name}.{model.short_id}_gold_n{n}.npy", gold_embeddings)
        np.save(OUT_DIR / f"{file_name}.{model.short_id}_gen_n{n}.npy", gen_embeddings)
