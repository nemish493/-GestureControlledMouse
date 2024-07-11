import cv2
import mediapipe as mp
import pyautogui
import random
import numpy as np
from pynput.mouse import Button, Controller

# Initialize mouse controller
mouse = Controller()

# Get the screen width and height
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Function to calculate the angle between three points
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

# Function to calculate the distance between two points, normalized to a range of 0 to 1000
def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return 0
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

# Function to find the tip of the index finger
def get_index_finger_tip(processed_hands):
    if processed_hands.multi_hand_landmarks:
        hand_landmarks = processed_hands.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return index_tip
    return None

# Function to move the mouse pointer based on the index finger tip position
def move_cursor(index_tip):
    if index_tip is not None:
        x_coord = int(index_tip.x * screen_w)
        y_coord = int(index_tip.y / 2 * screen_h)
        pyautogui.moveTo(x_coord, y_coord)

# Functions to determine the gesture for left click, right click, double click, and screenshot
def check_left_click(landmarks, thumb_index_distance):
    return (
        get_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        get_angle(landmarks[9], landmarks[10], landmarks[12]) > 90 and
        thumb_index_distance > 50
    )

def check_right_click(landmarks, thumb_index_distance):
    return (
        get_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        get_angle(landmarks[5], landmarks[6], landmarks[8]) > 90 and
        thumb_index_distance > 50
    )

def check_double_click(landmarks, thumb_index_distance):
    return (
        get_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        get_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        thumb_index_distance > 50
    )

def check_screenshot(landmarks, thumb_index_distance):
    return (
        get_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        get_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        thumb_index_distance < 50
    )

# Function to detect gestures and perform actions accordingly
def recognize_gesture(frame, landmarks, processed_hands):
    if len(landmarks) >= 21:
        index_tip = get_index_finger_tip(processed_hands)
        thumb_index_dist = get_distance([landmarks[4], landmarks[5]])

        if thumb_index_dist < 50 and get_angle(landmarks[5], landmarks[6], landmarks[8]) > 90:
            move_cursor(index_tip)
        elif check_left_click(landmarks, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif check_right_click(landmarks, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif check_double_click(landmarks, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif check_screenshot(landmarks, thumb_index_dist):
            screenshot = pyautogui.screenshot()
            label = random.randint(1, 1000)
            screenshot.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# Main function to capture video and process each frame
def main():
    draw_utils = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_hands = hand_tracker.process(frame_rgb)

            landmark_points = []
            if processed_hands.multi_hand_landmarks:
                hand_landmarks = processed_hands.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_points.append((lm.x, lm.y))

            recognize_gesture(frame, landmark_points, processed_hands)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Entry point of the script
if __name__ == '__main__':
    main()
