import sys
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.json_pp_iterator import iterator
from src.model_info import MODEL_INFO

DATA_DIR = Path(__file__).parent.parent / "main_paper_data" / "data"
OUT_DIR = Path(__file__).parent.parent / "data" / "embeddings"


def text_to_embeddings(text: str, tokenizer, model):
    max_length=512
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunk_size = max_length - 2
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        ids = torch.tensor([[tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]])
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        chunks.append(out[0][0, 1:-1].cpu().numpy())
    return np.concatenate(chunks, axis=0)


OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-group entries by file to avoid re-iterating per model
by_file = {}
for file_name, prefix, gold_completion, gen_completion in iterator(DATA_DIR):
    if isinstance(gen_completion, str):
        by_file.setdefault(file_name, []).append((gold_completion, gen_completion))

for model_name, info in MODEL_INFO.items():
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    for file_name, entries in by_file.items():
        n = len(entries)
        print(f"  {file_name} has {n} entries")

        gold_embeddings = np.empty(n, dtype=object)
        gen_embeddings = np.empty(n, dtype=object)

        for i, (gold, gen) in enumerate(entries):
            gold_embeddings[i] = text_to_embeddings(gold, tokenizer, model)
            gen_embeddings[i] = text_to_embeddings(gen, tokenizer, model)

        short_id = info["short_id"]
        np.save(OUT_DIR / f"{file_name}.{short_id}_gold_n{n}.npy", gold_embeddings)
        np.save(OUT_DIR / f"{file_name}.{short_id}_gen_n{n}.npy", gen_embeddings)
