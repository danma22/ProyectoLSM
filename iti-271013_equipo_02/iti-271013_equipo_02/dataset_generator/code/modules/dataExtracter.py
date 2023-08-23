import numpy as np
import math
import pandas as pd
import os
import cv2
import mediapipe as mp
import time


class landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def calculate_distance(base_of_hand: np.array, list_of_points: np.array) -> np.array:
    """This functions returns an array with all the distances from the base_of_hand to all entries in list_of_points.

    Args:
        base_of_hand (np.array): array of size n.
        list_of_points (np.array): array of size n*3.

    Returns:
        np.array: array of size n.
    """
    return np.array([math.dist(base_of_hand, p) for p in list_of_points])


def calculate_angles(base_of_hand: np.array, list_of_points: np.array) -> np.array:
    """
    This functions returns an array with all the angles from the base_of_hand to all entries in list_of_points.

    Args:
            base_of_hand (np.array): array of size n.
            list_of_points (np.array): array of size n*3.

    Returns:
            np.array: array of size n.
    """
    return np.array(
        [
            math.degrees(math.atan2(p[1] - base_of_hand[1], p[0] - base_of_hand[0]))
            for p in list_of_points
        ]
    )


def calculate_middle_finger_distance(list_of_points: np.array) -> np.array:
    """
        This functions returns an array with all the distances between subsequent points in the array.

    Args:
        list_of_points (np.array): array of size n.

    Returns:
        np.array: array of size n.
    """
    dist = 0
    for i in range(len(list_of_points) - 1):
        dist += math.dist(list_of_points[i], list_of_points[i + 1])
    return dist


def prepare_workspace(folders: list[str]) -> None:
    import shutil

    shutil.rmtree("../../dataset")

    import os

    os.mkdir("../../dataset")
    for letter in folders:
        os.mkdir("../../dataset/" + letter)


def main():

    letters = os.listdir("../../original_dataset/Hands")
    for letter in letters:
        for iteration in range(
            1, len(os.listdir("../../original_dataset/Hands/" + letter))
        ):
            raw_data = pd.read_csv(
                "../../original_dataset/Hands/"
                + letter
                + "/"
                + letter
                + str(iteration)
                + ".txt",
                header=None,
            )
            indexlessData = (raw_data.iloc[:, 2:]).copy()

            indexlessData = indexlessData.to_numpy()

            # print(indexlessData.shape)
            # print(indexlessData[0].shape)

            data_l = []
            data_r = []
            landmarks = []

            for i in range(0, 72):
                if i % 2 != 0:
                    aux = indexlessData[i].reshape(21, 3)
                    for j in range(21):
                        landmarks.append(landmark(aux[0], aux[1], aux[2]))

                    # print(aux)

                    left_hand_base = landmarks[0]
                    left_hand_base = np.array([aux[0][0], aux[0][1], aux[0][2]])

                    # Salto debería de ser 3?
                    left_hands_data_points = aux[1::2]
                    left_hand_landmarks = left_hands_data_points

                    middle_finger_points = list(aux[9:13])
                    middle_finger_points.insert(0, aux[0])
                    # middle_finger_points = [
                    # np.array([p.x, p.y, p.z]) for p in middle_finger_points
                    # ]
                    middle_finger_points = np.array(middle_finger_points)

                    # print(middle_finger_points.shape)
                    m_f_d = calculate_middle_finger_distance(middle_finger_points)
                    l_d = calculate_distance(left_hand_base, left_hand_landmarks)
                    l_a = calculate_angles(left_hand_base, left_hand_landmarks)
                    data_l.append((l_d * m_f_d, l_a))
                else:
                    aux = indexlessData[i].reshape(21, 3)
                    for j in range(21):
                        landmarks.append(landmark(aux[0], aux[1], aux[2]))

                    # print(aux)

                    right_hand_base = landmarks[0]
                    right_hand_base = np.array([aux[0][0], aux[0][1], aux[0][2]])

                    right_hands_data_points = aux[1::2]
                    right_hand_landmarks = right_hands_data_points

                    middle_finger_points = list(aux[9:13])
                    middle_finger_points.insert(0, aux[0])
                    # middle_finger_points = [
                    # np.array([p.x, p.y, p.z]) for p in middle_finger_points
                    # ]
                    middle_finger_points = np.array(middle_finger_points)

                    # print(middle_finger_points.shape)
                    m_f_d = calculate_middle_finger_distance(middle_finger_points)
                    r_d = calculate_distance(right_hand_base, right_hand_landmarks)
                    r_a = calculate_angles(right_hand_base, right_hand_landmarks)
                    data_r.append((r_d * m_f_d, r_a))
            with open(
                f"../../original_dataset/data/{letter}/{letter}_{(iteration)}.csv",
                "w",
            ) as f:
                for o in range(len(data_l)):

                    data_l[o][0].tofile(f, sep=",", format="%s")
                    f.write(",")
                    data_l[o][1].tofile(f, sep=",", format="%s")
                    f.write(",")
                    data_r[o][0].tofile(f, sep=",", format="%s")
                    f.write(",")
                    data_r[o][1].tofile(f, sep=",", format="%s")

                    f.write("\n")

            data_l = []
            data_r = []

    # print(data_r)
    # print(data_l)
    # print(letters)


# To convert index-less data to form array 21x3. Es un reshape xdddddddddddddddddddddddd


def extract_from_video():
    framerate = 12
    frame_limit = 36
    letters = ["a","b","l","i","j","ñ","encender","gracias","f"]
    number_of_extractions = 3

    prepare_workspace(letters)

    data_l = []
    data_r = []

    mp_holistic = mp.solutions.holistic

    video_folder = "../../../dataset_generator/videos/LSM Dataset/"

    for place in os.listdir(video_folder):
        place_folder = os.path.join(video_folder, place)
        
        if os.path.isdir(place_folder):
            for videos in os.listdir(place_folder):
                if videos.endswith(".avi"):
                    video_path = os.path.join(place_folder, videos)
                    cap = cv2.VideoCapture(video_path)
            # set framerate to 12
            cap.set(cv2.CAP_PROP_FPS, framerate)

            actual_frame = 0
            actual_extraction = 0
            actual_sign = 0

            with mp_holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as holistic:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        break

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)

                    if results.left_hand_landmarks is not None:
                        left_hand_base = results.left_hand_landmarks.landmark[0]
                        left_hand_base = np.array(
                            [left_hand_base.x, left_hand_base.y, left_hand_base.z]
                        )

                        left_hands_data_points = results.left_hand_landmarks.landmark[
                            1::2
                        ]
                        left_hand_landmarks = [
                            np.array([p.x, p.y, p.z]) for p in left_hands_data_points
                        ]

                        middle_finger_points = results.left_hand_landmarks.landmark[
                            9:13
                        ]
                        middle_finger_points.insert(
                            0, results.left_hand_landmarks.landmark[0]
                        )
                        middle_finger_points = [
                            np.array([p.x, p.y, p.z]) for p in middle_finger_points
                        ]

                        # print(calculate_distance(left_hand_base, left_hand_landmarks))
                        # print(calculate_angles(left_hand_base, left_hand_landmarks))
                        m_f_d = calculate_middle_finger_distance(middle_finger_points)
                        l_d = calculate_distance(left_hand_base, left_hand_landmarks)
                        l_a = calculate_angles(left_hand_base, left_hand_landmarks)
                        data_l.append((l_d * m_f_d, l_a))

                    else:
                        data_l.append((np.zeros(10), np.zeros(10)))

                    if results.right_hand_landmarks is not None:
                        right_hand_base = results.right_hand_landmarks.landmark[0]
                        right_hand_base = np.array(
                            [right_hand_base.x, right_hand_base.y, right_hand_base.z]
                        )

                        right_hands_data_points = results.right_hand_landmarks.landmark[
                            1::2
                        ]
                        right_hand_landmarks = [
                            np.array([p.x, p.y, p.z]) for p in right_hands_data_points
                        ]

                        middle_finger_points = results.right_hand_landmarks.landmark[
                            9:13
                        ]
                        middle_finger_points.insert(
                            0, results.right_hand_landmarks.landmark[0]
                        )
                        middle_finger_points = [
                            np.array([p.x, p.y, p.z]) for p in middle_finger_points
                        ]

                        # print(calculate_distance(right_hand_base, right_hand_landmarks))
                        # print(calculate_angles(right_hand_base, right_hand_landmarks))
                        m_f_d = calculate_middle_finger_distance(middle_finger_points)
                        r_d = calculate_distance(right_hand_base, right_hand_landmarks)
                        r_a = calculate_angles(right_hand_base, right_hand_landmarks)
                        data_r.append((r_d * m_f_d, r_a))
                    else:
                        data_r.append((np.zeros(10), np.zeros(10)))

                    actual_frame += 1
                    print(
                        "Recording letter {} Frame {} Extraction {}".format(
                            letters[actual_sign], actual_frame, actual_extraction
                        )
                    )
                    if actual_frame == frame_limit:
                        if actual_extraction == number_of_extractions:
                            if actual_sign == len(letters):
                                break
                            print("Starting new letter {}".format(letters[actual_sign]))
                            actual_sign += 1
                            actual_extraction = 0
                            actual_frame = 0
                            time.sleep(3)

                            data_l = []
                            data_r = []
                        else:
                            actual_extraction += 1
                            actual_frame = 0

                            print("Owo")
                            with open(
                                f"../../dataset/{place}/{videos}.csv",
                                "w",
                            ) as f:
                                for i in range(len(data_l)):

                                    data_l[i][0].tofile(f, sep=",", format="%s")
                                    f.write(",")
                                    data_l[i][1].tofile(f, sep=",", format="%s")
                                    f.write(",")
                                    data_r[i][0].tofile(f, sep=",", format="%s")
                                    f.write(",")
                                    data_r[i][1].tofile(f, sep=",", format="%s")

                                    f.write("\n")

                            data_l = []
                            data_r = []

                    # Draw landmark annotation on the image.
                    image.flags.writeable = False
                cap.release()


if __name__ == "__main__":
    main()
    #extract_from_video()
    # restructuredData.to_csv("out2.csv",index=None)
    pass