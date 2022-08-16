import pandas as pd
import pytest
import os
import tqdm

def test_data_all_exists():

    url = 'https://drive.google.com/file/d/1c_BEjy302XTqOEzTh1ZOFUsDhrxQt5BQ/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    print("Downloading csv")
    df = pd.read_feather(path)

    for path in tqdm.tqdm(df["Images"]):

        assert os.path.exists(path), f"File {Path} can't be found"

if __name__ == "__main__":

    test_data_all_exists()