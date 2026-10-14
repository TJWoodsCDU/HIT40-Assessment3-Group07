import sys
import config as C
import response_eda
import merge_datasets
import eda
import models


def main():
    needed = ["matches.csv", "teams.csv", "form.csv"]
    missing = [f for f in needed if not (C.DATA_DIR / f).exists()]
    if missing:
        sys.exit(f"Missing {missing} in {C.DATA_DIR}. Run Parts 1 to 3 first, then run again.")
    response_eda.main()      # needs only matches.csv
    merge_datasets.main()    # builds the two datasets and checks them
    eda.main()
    models.main()


if __name__ == "__main__":
    main()