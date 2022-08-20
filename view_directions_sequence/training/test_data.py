import pandas as pd
import numpy as np


def test_all_data_avaialble():

    df = pd.read_feather("training.df")
    all_keys = np.sort(np.unique(np.concatenate(df["keys"])))

    saved = pd.read_feather("../seq_data/data.feather")
    print(saved["key"].isna().any())
    saved_keys = np.sort(saved["key"].values)

    assert np.all(all_keys == saved_keys)
