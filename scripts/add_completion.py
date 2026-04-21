import os
import sqlite3

from openai import OpenAI
from openrouter import OpenRouter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "completions.db"

model_name = "gpt-5.4-mini"
open_router_model_id = "gpt-5"
openai_model_id = "gpt-5.4-mini"

import sys
sys.path.insert(0, str(ROOT))
from src.db_iterator import db_iterator

def _query_open_router(api_key: str, prefix_text: str) -> str:
    with OpenRouter(api_key) as client:
        response = client.chat.send(
            model = open_router_model_id,
            messages=[
                {
                    "role": "user",
                    "content": f"Continue this text: {prefix_text}"
                }
            ],
            max_tokens=300,  # To match paper's ~300 tokens
            temperature=1.0
    )

    return response.choices[0].message.content.strip()

def _query_openai(prefix_text: str) -> str:
    with OpenAI() as client:
        response = client.responses.create(
            model = openai_model_id,
            max_output_tokens=300,
            reasoning={"effort": "none"},
            input=[
                {
                    "role": "user",
                    "content": f"Continue this text: {prefix_text}"
                }
            ]
        )

    return response.output_text

def _get_prefix_id(cursor, prefix_text, source) -> int:
    cursor.execute("SELECT id FROM prefixes WHERE text = ? AND source = ?", (prefix_text, source))
    row = cursor.fetchone()
    assert row is not None, f"No prefix found for text: {prefix_text} and source: {source}"
    return int(row[0])

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    api_key = os.getenv("OPEN_ROUTER_KEY")
    assert api_key, "OPEN_ROUTER_KEY environment variable not set"

    for source, prefix_text, _, _ in db_iterator(str(DB_PATH)):
        prefix_id = _get_prefix_id(cursor, prefix_text, source)

        # avoid duplicates
        cursor.execute("SELECT id FROM completions WHERE model = ? AND prefix_id = ?", (model_name, prefix_id))
        if cursor.fetchone():
            continue

        # generate new completion
        try:
            # completion_text = _query_open_router(api_key, prefix_text)
            completion_text = _query_openai(prefix_text)
            word_count = len(completion_text.split())

            cursor.execute(
                "INSERT INTO completions (model, text, word_count, prefix_id) VALUES (?, ?, ?, ?)",
                (model_name, completion_text, word_count, prefix_id)
            )

        except Exception as e:
            print(f"Generation error prefix_id={prefix_id} prefix_text={prefix_text[:60]} ...: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()