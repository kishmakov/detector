import sqlite3


def db_iterator(db_path):
    conn = sqlite3.connect(db_path)
    cur_prefixes = conn.cursor()

    # Get all prefixes
    cur_prefixes.execute("SELECT id, text, source FROM prefixes")
    for prefix_id, prefix_text, source in cur_prefixes.fetchall():
        # Get human completion
        cur_completions = conn.cursor()
        cur_completions.execute("SELECT text FROM completions WHERE prefix_id = ? AND model = 'human'", (prefix_id,))
        human_row = cur_completions.fetchone()
        assert human_row, f"No human completion found for prefix_id={prefix_id}"
        human_completion = human_row[0]

        # Get machine completions
        cur_completions.execute("SELECT model, text FROM completions WHERE prefix_id = ? AND model != 'human'", (prefix_id,))
        machine_completions = [(row[0], row[1]) for row in cur_completions.fetchall()]

        yield source, prefix_text, human_completion, machine_completions

    conn.close()