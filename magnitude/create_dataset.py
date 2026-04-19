import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from json_pp_iterator import iterator

DATA_DIR = "../main_paper_data/data"

for i, (file_name, prefix, gold_completion, gen_completion) in enumerate(iterator(DATA_DIR)):
    print(f"--- {file_name} [{i}] ---")
    print(f"GOLD: {gold_completion[:120].strip()}")
    print(f"GEN:  {gen_completion[0][:120].strip()}\n")
