"""
run_all.py - runs the whole Part 4 pipeline in order.

    python run_all.py dummy     # test everything on fake data (creates the fake files first)
    python run_all.py real      # use the real files in data/real/

For 'real' you must first put matches.csv, teams.csv and form.csv into data/real/.
"""
import sys
import config as C
import response_eda
import merge_datasets
import eda
import models


def main(mode):
    if mode == "dummy":
        import runpy
        runpy.run_path(str(C.HERE / "make_dummy_data.py"))
    else:
        needed = ["matches.csv", "teams.csv", "form.csv"]
        missing = [f for f in needed if not (C.data_dir(mode) / f).exists()]
        if missing:
            sys.exit(f"Missing {missing} in {C.data_dir(mode)}. Put the three files there and run again.")
    response_eda.main(mode)   # needs only matches.csv
    merge_datasets.main(mode)
    eda.main(mode)
    models.main(mode)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dummy")