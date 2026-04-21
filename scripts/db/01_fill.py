import sqlite3
import sys

from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "completions.db"

sys.path.insert(0, str(ROOT))
from src.json_iterator import json_iterator


def _extract_model(filename: str) -> str:
    """e.g., 'human_gpt3_davinci_003_reddit' -> 'gpt3'"""
    return filename.split('_')[1]


def _extract_source(filename: str) -> str:
    """e.g., 'human_gpt2_wikip' -> 'wiki'"""
    source = filename.split('_')[-1]
    return 'wiki' if source == 'wikip' else source


def _get_or_create_prefix(conn: sqlite3.Connection, text: str, source: str) -> int:
    """Get prefix id or create if not exists."""
    text = text.strip()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM prefixes WHERE text = ? AND source = ?", (text, source))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("INSERT INTO prefixes (text, source) VALUES (?, ?)", (text, source))
    return cursor.lastrowid


def _insert_completion(conn: sqlite3.Connection, model: str, text: str, prefix_id: int) -> None:
    """Insert a completion into the database."""
    text = text.strip()
    word_count = len(text.split())
    conn.execute(
        "INSERT INTO completions (model, text, word_count, prefix_id) VALUES (?, ?, ?, ?)",
        (model, text, word_count, prefix_id)
    )


def main() -> None:
    data_dir = ROOT / "data"

    old_source = ""
    old_model = ""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM completions")
        conn.execute("DELETE FROM prefixes")

        for filename, prefix_text, gold_completion, gen_completion in tqdm(json_iterator(data_dir)):
            source = _extract_source(filename)
            model = _extract_model(filename)

            if source != old_source or model != old_model:
                print(f"Processing source={source} model={model}")
                old_source = source
                old_model = model

            prefix_id = _get_or_create_prefix(conn, prefix_text, source)

            _insert_completion(conn, "human", gold_completion, prefix_id)

            if isinstance(gen_completion, str):
                gen_completion = [gen_completion]

            for comp in gen_completion:
                _insert_completion(conn, model, comp, prefix_id)

        conn.commit()


if __name__ == "__main__":
    main()

