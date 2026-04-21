import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "completions.db"
SCHEMA_PATH = ROOT / "data" / "migrations" / "01_init.sql"


def main() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
            conn.executescript(schema_file.read())

    print(f"Initialized database: {DB_PATH}")


if __name__ == "__main__":
    main()
