import sqlite3


def db_iterator(db_path, *, source=None, model=None):
    """Arguments:
        db_path: Path to the SQLite database.
        source: in [None, 'wiki', 'reddit']
        model: in [None, 'gemini-3.1-pro', 'gpt-5.4-mini', 'gpt2', 'gpt3', 'opt13']
    """
    conn = sqlite3.connect(db_path)

    prefixes_queries = "SELECT id, text, source FROM prefixes"
    prefixes_params = []
    if source is not None:
        prefixes_queries += " WHERE source = ?"
        prefixes_params.append(source)

    prefixes_cur = conn.cursor()
    prefixes_cur.execute(prefixes_queries, prefixes_params)
    for prefix_id, prefix_text, source in prefixes_cur.fetchall():
        # Get human completion
        completions_cur = conn.cursor()
        completions_cur.execute("SELECT text FROM completions WHERE prefix_id = ? AND model = 'human'", (prefix_id,))
        human_row = completions_cur.fetchone()
        assert human_row, f"No human completion found for prefix_id={prefix_id}"
        human_completion = human_row[0]

        # Get machine completions
        completions_query = "SELECT model, text FROM completions WHERE prefix_id = ? AND model != 'human'"
        completions_params = [prefix_id]
        if model is not None:
            completions_query += " AND model = ?"
            completions_params.append(model)
        completions_cur.execute(completions_query, completions_params)
        machine_completions = [(row[0], row[1]) for row in completions_cur.fetchall()]

        yield source, prefix_text, human_completion, machine_completions

    conn.close()