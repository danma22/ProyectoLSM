import time
import cv2
import mediapipe as mp
import numpy as np
import csv
import math
import json

import tensorflow as tf
from tensorflow import keras


def calculate_distance(base_of_hand: np.array, list_of_points: np.array) -> np.array:
    """Esta funcion regresa un numpy array de todas distancias de list_of_point hacia la base de la mano

    Args:
        base_of_hand (np.array): array de tamaño n
        list_of_points (np.array): array de tamaño n*3

    Returns:
        np.array: array de tamaño n
    """
    return np.array([math.dist(base_of_hand, p) for p in list_of_points])


def calculate_angles(base_of_hand: np.array, list_of_points: np.array) -> np.array:
    return np.array(
        [
            math.degrees(math.atan2(p[1] - base_of_hand[1], p[0] - base_of_hand[0]))
            for p in list_of_points
        ]
    )


def calculate_middle_finger_distance(list_of_points: np.array) -> np.array:
    dist = 0
    for i in range(len(list_of_points) - 1):
        dist += math.dist(list_of_points[i], list_of_points[i + 1])

    return dist


def format_data(data, data_l, data_r, framerate_limit=36):
    formatted_data = []
    for i in range(len(data_l)):
        data.append(
            np.concatenate(
                (data_l[i][0], data_l[i][1], data_r[i][0], data_r[i][1]), axis=0
            )
        )
    formatted_data = data

    if len(formatted_data) > framerate_limit:
        formatted_data = formatted_data[-framerate_limit:]

    return formatted_data


def main():
    framerate = 24
    frame_limit = 36

    data = []
    data_l = []
    data_r = []

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    model = keras.models.load_model("best_model_acc94_150_e.h5")
    print(model.summary())

    with open("le_name_mapping.txt", "r") as f:
        x = f.read()

    name_mapping = json.loads(x)

    cap = cv2.VideoCapture(0)
    # set framerate to 12
    cap.set(cv2.CAP_PROP_FPS, framerate)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        pre_text = ""
        text = ""
        porcentage_pred = 0.0

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

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

                left_hands_data_points = results.left_hand_landmarks.landmark[1::2]
                left_hand_landmarks = [
                    np.array([p.x, p.y, p.z]) for p in left_hands_data_points
                ]

                middle_finger_points = results.left_hand_landmarks.landmark[9:13]
                middle_finger_points.insert(0, results.left_hand_landmarks.landmark[0])
                middle_finger_points = [
                    np.array([p.x, p.y, p.z]) for p in middle_finger_points
                ]

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

                right_hands_data_points = results.right_hand_landmarks.landmark[1::2]
                right_hand_landmarks = [
                    np.array([p.x, p.y, p.z]) for p in right_hands_data_points
                ]

                middle_finger_points = results.right_hand_landmarks.landmark[9:13]
                middle_finger_points.insert(0, results.right_hand_landmarks.landmark[0])
                middle_finger_points = [
                    np.array([p.x, p.y, p.z]) for p in middle_finger_points
                ]

                m_f_d = calculate_middle_finger_distance(middle_finger_points)
                r_d = calculate_distance(right_hand_base, right_hand_landmarks)
                r_a = calculate_angles(right_hand_base, right_hand_landmarks)
                data_r.append((r_d * m_f_d, r_a))
            else:
                data_r.append((np.zeros(10), np.zeros(10)))

            # print(len(data_l))
            # if len(data_l) < frame_limit:

            data = format_data(data, data_l, data_r)
            data_formatted = np.array(data)
            prediction_text = ""

            if data_formatted.shape[0] == frame_limit:
                # print(data_formatted.flatten().shape)
                # print(data_formatted.flatten().reshape(1, -1, 1).shape)
                result = model.predict(data_formatted.flatten().reshape(1, -1, 1), verbose = 0)
                prediction_text = f"{name_mapping[str(np.argmax(result))]} {result[0][np.argmax(result)]:2.3f}"
                
                porcentage_pred = result[0][np.argmax(result)]
                text = name_mapping[str(np.argmax(result))]
                 
                # it only shows the new expression prediction
                if pre_text != text and text != 'none':
                    print(
                        np.argmax(result),
                        name_mapping[str(np.argmax(result))],
                        # result[0][np.argmax(result)],
                    )

                pre_text = text
                

            # Draw landmark annotation on the image.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            # for hand_no,hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)

            # put in screen the name of the gesture and the confidence
            if (text != 'none'):
                cv2.putText(
                    image,
                    prediction_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("MediaPipe Holistic", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
