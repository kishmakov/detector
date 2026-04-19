import json
from pathlib import Path


def iterator(directory):
    for path in sorted(Path(directory).glob("*.json_pp")):
        with open(path) as f:
            for entry in json.load(f):
                yield path.name.replace(".json_pp", ""), entry["prefix"], entry["gold_completion"], entry["gen_completion"]
