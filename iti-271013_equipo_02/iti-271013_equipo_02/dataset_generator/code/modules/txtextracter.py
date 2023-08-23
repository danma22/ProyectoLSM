import time
import cv2
import mediapipe as mp
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

final_face_data = []
final_pose_data = []
final_hand_data = []
word_bank = [
    # AQUI VAN LAS PALABRAS A CAPTURAR 
    "encender"
]
data = []
word_list = [
    "encender"
]
currentPosition = 0
word = word_list[currentPosition]
iteration = 0


# https://google.github.io/mediapipe/solutions/holistic


def record_data() -> None:
    """Saves raw data to the appropiate location"""
    global iteration, final_hand_data, final_face_data, final_pose_data, currentPosition, word
    print("trying to print " + word + str(iteration))
    with open("../../original_dataset/Hands/" + word + "/" + word + str(iteration) + ".txt", "w+") as f:
        write = csv.writer(f)
        write.writerows(final_hand_data)
    with open(
        "../../original_dataset/Face/" + word + "/" + word + str(iteration) + ".txt", "w+"
    ) as f:
        write = csv.writer(f)
        write.writerows(final_face_data)
    with open("../../original_dataset/Torso/" + word + "/" + word + str(iteration) + ".txt", "w+") as f:
        write = csv.writer(f)
        write.writerows(final_pose_data)
    final_hand_data = []
    iteration += 1
    word = word_list[currentPosition]
    if iteration == len(get_ready_words(f"../../videos/LSM dataset/{word}/")) + 11:
        iteration = len(get_ready_words(f"../../videos/LSM dataset/{word}/"))
        currentPosition += 1

    final_face_data = []
    final_pose_data = []


def get_ready_words(directory: str) -> None:
    """Get file names from the current directory"""
    existing_words = os.listdir(directory)
    return existing_words


def live_data_extracter() -> None:
    """Uses webcam feed to extract both hands coordinates as described in this image: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png"""
    global final_hand_data, final_face_data, final_pose_data, currentPosition, word, iteration
    hand_data_frameR = []
    face_data = []
    hand_data_frameL = []
    pose_data = []
    frame = 0
    frame_rate = 12
    prev = 0
    begin = False
    end = False
    iteration = len(get_ready_words(f"../../videos/LSM dataset/{word}/"))
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            time_elapsed = time.time() - prev
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            if time_elapsed > 1.0 / frame_rate:
                prev = time.time()

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw landmark annotation on the image.
                image.flags.writeable = True
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

                if begin == True:
                    frame += 1

                    hand_data_frameR.append(str(frame))
                    hand_data_frameR.append("right")
                    if results.right_hand_landmarks is not None:
                        for hand_landmarks in results.right_hand_landmarks.landmark:
                            hand_data_frameR.append(str(hand_landmarks.x))
                            hand_data_frameR.append(str(hand_landmarks.y))
                            hand_data_frameR.append(str(hand_landmarks.z))
                        # final_hand_data.append(hand_data_frameR)
                        # hand_data_frameR = []
                    else:
                        for i in range(63):
                            hand_data_frameR.append(0)
                    final_hand_data.append(hand_data_frameR)
                    hand_data_frameR = []

                    hand_data_frameL.append(str(frame))
                    hand_data_frameL.append("left")
                    if results.left_hand_landmarks is not None:
                        for hand_landmarks in results.left_hand_landmarks.landmark:
                            hand_data_frameL.append(str(hand_landmarks.x))
                            hand_data_frameL.append(str(hand_landmarks.y))
                            hand_data_frameL.append(str(hand_landmarks.z))

                        # final_hand_data.append(hand_data_frameL)
                        # hand_data_frameL = []
                    else:
                        for i in range(63):
                            hand_data_frameL.append(0)
                    final_hand_data.append(hand_data_frameL)
                    hand_data_frameL = []

                    for i in range(0, 11):
                        face_data.append(str(results.pose_landmarks.landmark[i].x))
                        face_data.append(str(results.pose_landmarks.landmark[i].y))
                        face_data.append(str(results.pose_landmarks.landmark[i].z))
                    final_face_data.append(face_data)
                    face_data = []
                    for i in range(11, 13):
                        pose_data.append(str(results.pose_landmarks.landmark[i].x))
                        pose_data.append(str(results.pose_landmarks.landmark[i].y))
                        pose_data.append(str(results.pose_landmarks.landmark[i].z))
                    final_pose_data.append(pose_data)
                    pose_data = []
                    if frame == 36:
                        # if end == True:
                        frame = 0
                        begin = False
                        # end = False
                        record_data()
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow("MediaPipe Holistic", cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == ord("a"):
                    if begin == False:
                        print(word)
                        begin = True
                    # else:
                    #   end = True

    cap.release()


def unique_values(list_a: list, list_b: list) -> set:
    """
    Compares to lists and gets the values in one but not in the other.

    Args:

        list_a (list): Base list to compare.
        list_b (list): Elements in second list to substract.


    Returns:
      set: Set containing values that exist in the first list but not the second.
    """
    a = set(list_a)
    b = set(list_b)

    return (a - b)

def getParent(path, levels):
    common = path
 
    # Using for loop for getting
    # starting point required for
    # os.path.relpath()
    for i in range(levels + 1):
 
        # Starting point
        common = os.path.dirname(common)
 
    # Parent directory upto specified
    # level
    return os.path.relpath(path, common)
 

    

def record_video() -> None:
    """Record a video and show the wireframe overlay"""
    global final_hand_data, final_face_data, final_pose_data, currentPosition, word

    frame = 0
    frame_rate = 12
    prev = 0
    index = 0
    begin = False
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    available_directories = get_ready_words("../../videos/LSM dataset/")

    for i in unique_values(word_bank, available_directories):
        os.mkdir("../../videos/LSM dataset/"+i)
    available_directories = get_ready_words("../../videos/LSM dataset/")
    folder = available_directories[index]

    print(available_directories)
    available_words = get_ready_words(f"../../videos/LSM dataset/{folder}/")

    iteration = len(available_words) + 1

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            time_elapsed = time.time() - prev
            success, image = cap.read()
            image_to_add = image

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            if time_elapsed > 1.0 / frame_rate:
                prev = time.time()

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw landmark annotation on the image.
                image.flags.writeable = True
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

                if begin == True:
                    frame += 1
                    out.write(image_to_add)
                    if frame == 36:
                        frame = 0
                        begin = False
                        out.release()
                        print(len(available_words))
                        if iteration == len(available_words) + 10:
                            index += 1
                            folder = available_directories[index]
                            available_words = get_ready_words(
                                f"../../videos/LSM dataset/{folder}/"
                            )
                            iteration = len(available_words)
                        iteration += 1
                        print("Finished Recording")

                cv2.imshow("MediaPipe Holistic", cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == ord("a"):
                    if begin == False:
                        begin = True
                        print(f"Recording {word}{iteration}")
                        out = cv2.VideoWriter(
                            f"../../videos/LSM dataset/{available_directories[index]}/{word}{iteration}.avi",
                            cv2.VideoWriter_fourcc(*"DIVX"),
                            12,
                            (width, height),
                        )


if __name__ == "__main__":
    ##record_video()
    live_data_extracter()
    pass
