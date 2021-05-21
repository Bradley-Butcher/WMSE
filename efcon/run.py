from efcon.fast import score_file
import pandas as pd


if __name__ == "__main__":
    data = pd.read_csv("../data/child_data.csv")

    data = data.apply(lambda x: x.astype("category").cat.codes)


    score_file(data, "../results/child_drcd_test_1.txt", 1, use_bic=False)
    score_file(data, "../results/child_bic_test.txt", 1, use_bic=True)