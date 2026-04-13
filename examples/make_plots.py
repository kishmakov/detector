import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from magnitude.text_processing import preprocess_text


FIRST_SEGMENT_WORDS = 250

MODEL_INFO = {
    "roberta-base":               {"style": "-",                   "short_id": "robb"},
    "roberta-large":              {"style": "--",                  "short_id": "robl"},
    "bert-base-uncased":          {"style": "-.",                  "short_id": "berb"},
    "bert-large-uncased":         {"style": ":",                   "short_id": "berl"},
    "distilroberta-base":         {"style": (0, (5, 1)),           "short_id": "disbn"},
    "distilbert-base-uncased":    {"style": (0, (3, 1, 1, 1)),     "short_id": "disbu"},
    "microsoft/deberta-v3-base":  {"style": (0, (7, 2)),           "short_id": "debb"},
    "microsoft/deberta-v3-large": {"style": (0, (3, 2, 1, 2)),     "short_id": "debl"},
}

TEXT_INFO = {
    "00_ai_gpt_364":      {"style": "-"},
    "01_ai_gpt_think_521":{"style": "--"},
    "02_ai_gem_531":      {"style": "-."},
    "03_ai_gem_think_581":{"style": ":"},
    "04_ai_ds_378":       {"style": (0, (5, 1))},
    "05_ai_ds_think_258": {"style": (0, (3, 1, 1, 1))},
    "06_repetition_256":  {"style": (0, (1, 1))},
    "07_repetition_512":  {"style": (0, (5, 5))},
    "08_defoe_339":       {"style": (0, (3, 5, 1, 5))},
}


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
    assert text, "Text is empty"

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


def compute_magnitude_curve(model, tokenizer, text: str) -> tuple[np.ndarray, np.ndarray]:
    X = text_to_embeddings(text, tokenizer, model)

    # log-spaced t values
    t_values = np.logspace(-3, 2, 40)  # 0.001 → 100
    magnitudes = np.array([magnitude_at_t(X, t) for t in t_values])

    return t_values, magnitudes


def plot_magnitude_comparison(
    text_label: str,
    curves_by_model: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, (t_values, magnitudes) in curves_by_model.items():
        finite = np.isfinite(magnitudes)
        positive = finite & (magnitudes > 0)
        line_style = MODEL_INFO[model_name]["style"]

        axes[0].loglog(
            t_values[positive],
            magnitudes[positive],
            linestyle=line_style,
            label=model_name,
        )
        axes[1].semilogx(
            t_values[finite],
            magnitudes[finite],
            linestyle=line_style,
            label=model_name,
        )

    axes[0].set_xlabel("log t (scale)")
    axes[0].set_ylabel("log |tA| (magnitude)")
    axes[0].set_title(f"text={text_label} model=* (log-log)")
    axes[0].legend()

    axes[1].set_xlabel("log t (scale)")
    axes[1].set_ylabel("linear |tA| (magnitude)")
    axes[1].set_title(f"text={text_label} model=* (log-linear)")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_text_segment_comparison(
    model_name: str,
    curves_by_text: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for text_label, (t_values, magnitudes) in curves_by_text.items():
        finite = np.isfinite(magnitudes)
        positive = finite & (magnitudes > 0)
        line_style = TEXT_INFO[text_label]["style"]

        axes[0].loglog(
            t_values[positive],
            magnitudes[positive],
            linestyle=line_style,
            label=text_label,
        )
        axes[1].semilogx(
            t_values[finite],
            magnitudes[finite],
            linestyle=line_style,
            label=text_label,
        )

    axes[0].set_xlabel("log t (scale)")
    axes[0].set_ylabel("log |tA| (magnitude)")
    axes[0].set_title(f"text=* model={model_name} (log-log)")
    axes[0].legend()

    axes[1].set_xlabel("log t (scale)")
    axes[1].set_ylabel("linear |tA| (magnitude)")
    axes[1].set_title(f"text=* model={model_name} (log-linear)")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    texts_dir = Path(__file__).parent / "texts"
    plots_root = Path(__file__).parent / "plots"

    preprocessed = {
        stem: preprocess_text(
            (texts_dir / f"{stem}.txt").read_text(encoding="utf-8"),
            max_words=FIRST_SEGMENT_WORDS,
        )
        for stem in TEXT_INFO
    }

    curves_by_stem = {stem: {} for stem in TEXT_INFO}
    segment_curves_by_model = {model_name: {} for model_name in MODEL_INFO}

    for model_name in MODEL_INFO:
        print("Processing", model_name, "...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        for stem, (full_text, segment_text) in preprocessed.items():
            print("Processing", stem, "...")
            curves_by_stem[stem][model_name] = compute_magnitude_curve(model, tokenizer, full_text)

            if segment_text:
                segment_curves_by_model[model_name][stem] = compute_magnitude_curve(
                    model,
                    tokenizer,
                    segment_text,
                )

    for stem, curves_by_model in curves_by_stem.items():
        plot_magnitude_comparison(
            stem,
            curves_by_model,
            plots_root / f"{stem}.png",
        )

    for model_name, curves_by_text in segment_curves_by_model.items():
        model_id = MODEL_INFO[model_name]["short_id"]
        output_path = plots_root / f"{model_id}_all_segments.png"
        plot_text_segment_comparison(model_name, curves_by_text, output_path)
