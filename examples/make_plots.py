import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.magnitude import magnitude_at_t
from src.text_utils import preprocess_text
from src.model import model_iterator, MODEL_INFO

MODEL_INFO = dict(MODEL_INFO)

FIRST_SEGMENT_WORDS = 250

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


# def text_to_static_embeddings(text: str, tokenizer, model) -> np.ndarray:
#     assert text, "Text is empty"

#     # 1. Tokenize the whole thing at once
#     # No need for max_length or special tokens for static lookups
#     token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

#     # 2. Direct lookup in the embedding matrix
#     # We move to the model's device in case you're on GPU
#     token_ids = token_ids.to(model.device)

#     with torch.no_grad():
#         # Access the underlying weight matrix
#         static_embeddings = model.embeddings.word_embeddings(token_ids)

#     # 3. Clean up shape: [1, sequence_length, hidden_size] -> [sequence_length, hidden_size]
#     result = static_embeddings.squeeze(0).cpu().numpy()
#     return result


def compute_magnitude_curve(model, text: str) -> tuple[np.ndarray, np.ndarray]:
    X = model.text_to_embeddings(text)

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
    segment_curves_by_model = {}

    for model in model_iterator()   :
        print("Processing", model.name, "...")

        for stem, (full_text, segment_text) in preprocessed.items():
            print("Processing", stem, "...")
            curves_by_stem.setdefault(stem, {})[model.name] = compute_magnitude_curve(model, full_text)

            if segment_text:
                segment_curves_by_model.setdefault(model.name, {})[stem] = compute_magnitude_curve(
                    model,
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
