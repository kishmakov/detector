import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel


def magnitude_at_t(X: np.ndarray, t: float, reg: float = 1e-6) -> float:
    """
    Compute the magnitude |tA| for a set of embeddings X at scale t.

    Parameters
    ----------
    X   : (n, d) embedding matrix
    t   : scale parameter
    reg : small regularization for stability

    """
    D = cdist(X, X, metric='euclidean')  # Pairwise distances
    Z = np.exp(-t * D)
    Z += reg * np.eye(Z.shape[0])

    # Solve Z w = 1
    w = np.linalg.solve(Z, np.ones(Z.shape[0]))
    return float(w.sum())


def text_to_embeddings(text: str, tokenizer, model, max_length: int = 512) -> np.ndarray:
    text = text.replace('\n', ' ')

    # Tokenize without special tokens to avoid the full-sequence length warning
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunk_size = max_length - 2  # reserve 2 slots for CLS + SEP per chunk
    chunks = []

    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i+chunk_size]
        ids = torch.tensor([[tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]])
        mask = torch.ones_like(ids)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)

        # Drop CLS and SEP embeddings
        emb = out[0][0, 1:-1].cpu().numpy()
        chunks.append(emb)

    assert len(chunks) > 0, "No embeddings generated"
    return np.concatenate(chunks, axis=0)


def plot_magnitude(model, tokenizer, text: str, output_path: Path) -> None:
    X = text_to_embeddings(text, tokenizer, model)

    # log-spaced t values
    t_values = np.logspace(-3, 2, 40)  # 0.001 → 100
    magnitudes = np.array([magnitude_at_t(X, t) for t in t_values])

    finite = np.isfinite(magnitudes)
    positive = finite & (magnitudes > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(t_values[positive], magnitudes[positive])
    axes[0].set_xlabel("log t (scale)")
    axes[0].set_ylabel("log |tA| (magnitude)")
    axes[0].set_title("Magnitude function (log-log)")

    axes[1].semilogx(t_values[finite], magnitudes[finite])
    axes[1].set_xlabel("log t (scale)")
    axes[1].set_ylabel("linear |tA| (magnitude)")
    axes[1].set_title("Magnitude function (log-linear)")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    for model_name in [
        "roberta-base",
        "roberta-large",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilroberta-base",
        "distilbert-base-uncased",
        "microsoft/deberta-v3-base",
        "microsoft/deberta-v3-large",
    ]:
        print("Processing", model_name, "...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        for path in Path('examples').glob('*.txt'):
            print("Processing", path, "...")
            text = path.read_text(encoding='utf-8')
            output_path = path.with_name(path.stem + f'_{model_name.replace("/", "_")}.png')
            plot_magnitude(model, tokenizer, text, output_path)