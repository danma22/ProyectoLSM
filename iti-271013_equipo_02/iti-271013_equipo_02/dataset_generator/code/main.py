# base app for generating the dataset
# Authors: Agustin Zavala, Jose Avalos, Roberto Higuera, Jesus Quiñones

"""
Autores para el rentrenamiento: 
    César Aldahir Flores Gámez
    Osiel Alejandro Ordoñez Cruz
    Mauricio Hernandez Cepeda
    Daniel Eduardo Macias Estrada
    Lorena Marisol Romero Hernandez    
"""

import os
import shutil
from modules.data_loader import load_data_numpy
from modules.neural_network_training import train_nn

def copy_original_dataset() -> None:
    for folder in os.listdir("../original_dataset/data"):
        # create folder if it doesn't exist
        if not os.path.exists("../dataset/" + folder):
            os.mkdir("../dataset/" + folder)
        number_of_existing_files = len(
            os.listdir(f"../dataset/{folder}")
        )
        for file in os.listdir(f"../original_dataset/data/{folder}"):
            number_of_existing_files += 1
            shutil.copyfile(
                src=f"../original_dataset/data/{folder}/{file}",
                dst=f"../dataset/{folder}/{number_of_existing_files}.csv",
            )


def generate_dataset() -> None:
    print("generating the dataset")
    shutil.rmtree("../dataset")
    os.mkdir("../dataset")

    # check if videos folder exists
    if os.path.exists("../videos"):
        print("Generating txt from videos")

    if os.path.exists("../original_dataset/data"):
        print("Copying original dataset")
        copy_original_dataset()

        

    x_train, y_train = load_data_numpy("../dataset")  


def save_tflite_model() -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(
        tf.keras.models.load_model("../best_model.h5")
    )
    tflite_model = converter.convert()

    with open("../model.tflite", "wb") as f:
        f.write(tflite_model)


def main() -> None:
    option = input("1.- generate the dataset\n2.- train the model\n3.- exit\n>> ")

    while option != "3":
        if option == "1":
            generate_dataset()
        elif option == "2":
            train_nn()
            save_tflite_model()
        else:
            print("invalid option")
        option = input("1.- generate the dataset\n2.- train the model\n3.- exit\n>> ")


if __name__ == "__main__":
    main()
