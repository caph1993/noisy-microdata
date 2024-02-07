from pathlib import Path
import pandas as pd
import numpy as np


def load_human_height():
    rootDir = Path(__file__).parent
    while True:
        csv = rootDir / "Data" / "Human_Height" / "Galton_data.csv"
        if csv.exists():
            break
        if rootDir.parent == rootDir:
            raise FileNotFoundError()
        rootDir = rootDir.parent
    df = pd.read_csv(csv, sep=";")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.drop("Unnamed: 1", axis=1, inplace=True)
    df.drop(0, axis=0, inplace=True)
    df.drop("Unnamed: 6", axis=1, inplace=True)

    # into floats
    df["Unnamed: 4"] = df["Unnamed: 4"].astype(float)
    df["Unnamed: 2"] = df["Unnamed: 2"].astype(float)
    df["Unnamed: 3"] = df["Unnamed: 3"].astype(float)
    df["Unnamed: 5"] = df["Unnamed: 5"].astype(float)

    # rename
    df.rename(
        columns={
            "Unnamed: 2": "father",
            "Unnamed: 3": "mother",
            "Unnamed: 4": "gender",
            "Unnamed: 5": "child",
        },
        inplace=True,
    )
    return df
