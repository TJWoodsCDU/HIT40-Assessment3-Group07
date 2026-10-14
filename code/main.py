import subprocess
import sys
import time
from pathlib import Path

import config as C
import response_eda
import merge_datasets
import eda
import models

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent                      # part3.py reads data/raw/... relative to this folder

PREP_STEPS = [
    ("Part 1: matches.csv", "part1.py"),
    ("Part 3: form.csv", "part3.py"),   # needs matches.csv from Part 1, so it runs second
    ("Part 2: teams.csv", "part2.py"),
]

PART4_STEPS = [
    ("Part 4a: results and baselines", response_eda.main),
    ("Part 4b: merge and checks", merge_datasets.main),
    ("Part 4c: exploratory data analysis", eda.main),
    ("Part 4d: models", models.main),
]


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70, flush=True)


def run_script(label, script):
    banner(label)
    start = time.time()
    result = subprocess.run([sys.executable, str(CODE / script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nStopped: {script} failed (see the error above). Fix it and run main.py again.")
    print(f"[{label} finished in {time.time() - start:.1f} s]", flush=True)


def run_function(label, function):
    banner(label)
    start = time.time()
    function()
    print(f"[{label} finished in {time.time() - start:.1f} s]", flush=True)


def main():
    begin = time.time()

    if "--part4-only" not in sys.argv:
        for label, script in PREP_STEPS:
            run_script(label, script)

    missing = [f for f in ("matches.csv", "teams.csv", "form.csv") if not (C.DATA_DIR / f).exists()]
    if missing:
        sys.exit(f"Missing {missing} in {C.DATA_DIR}. Run without --part4-only to build them.")

    for label, function in PART4_STEPS:
        run_function(label, function)

    banner(f"All done in {time.time() - begin:.0f} s. Results are in {C.OUT_DIR}")


if __name__ == "__main__":
    main()