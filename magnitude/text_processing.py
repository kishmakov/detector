import unicodedata


def _replace_punctuation_with_spaces(text: str) -> str:
    return "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )


def preprocess_text(text: str, max_words: int | None = None) -> tuple[str, str]:
    cleaned = _replace_punctuation_with_spaces(text)
    words = cleaned.split()
    full = " ".join(words)
    segment = " ".join(words[:max_words]) if max_words is not None else full
    return full, segment