-- 01_init.sql

CREATE TABLE prefixes (
    id       INTEGER PRIMARY KEY,
    text     TEXT,
    source   TEXT NOT NULL  -- e.g. 'wiki', 'reddit'
);

CREATE TABLE completions (
    id         INTEGER PRIMARY KEY,
    model      TEXT NOT NULL, -- 'human' or id of the model like 'gpt2'
    text       TEXT NOT NULL,
    word_count INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    prefix_id  INTEGER,
    FOREIGN KEY(prefix_id) REFERENCES prefixes(id)
);

CREATE INDEX idx_completions_prefix ON completions(prefix_id);
CREATE INDEX idx_completions_model  ON completions(model);
