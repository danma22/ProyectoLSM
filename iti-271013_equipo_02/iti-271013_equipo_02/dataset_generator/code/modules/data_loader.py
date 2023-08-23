import pandas as pd
import numpy as np
import os
import random
from sklearn import preprocessing


def load_data_numpy(folder: str) -> np.ndarray:
    data = []
    labels = []


    for letter_folder in os.listdir(folder):
        for file in os.listdir(f"{folder}/{letter_folder}"):
            data.append(
                np.genfromtxt(f"{folder}/{letter_folder}/{file}", delimiter=",")
            )
            data[-1] = data[-1].flatten()
            labels.append(letter_folder)

    labels = np.array(labels)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    with open(f"{folder}/../le_name_mapping.txt", "w") as f:
        f.write(str(le_name_mapping))
    print("Mppging of LabelEncoder")
    print(le_name_mapping)

    return np.array(data), labels


if __name__ == "__main__":
    load_data_numpy("../../original_dataset/data/")
