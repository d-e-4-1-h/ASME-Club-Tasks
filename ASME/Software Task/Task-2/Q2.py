"""
Gesture-Based Mouse and System Control using OpenCV and MediaPipe.

This script captures video from a webcam, detects hand landmarks in real-time,
and translates specific hand gestures into system-wide controls, including:
- Mouse pointer movement and smoothing.
- Left-clicks, right-clicks, and scrolling.
- Window management (minimize, maximize).

All key parameters like sensitivity and smoothing are configurable via an
external 'config.ini' file.
"""

import cv2
import mediapipe as mp
import time
import math
import pyautogui
import sys
import configparser
import numpy as np

# --- 1. CONFIGURATION AND INITIALIZATION ---

# Load settings from the config.ini file for easy tuning.
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize MediaPipe Hands solution.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0  # Use the lighter, faster model for better performance.
)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture and set a lower resolution for faster processing.
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 2. CONSTANTS AND GLOBAL STATE VARIABLES ---

# Load parameters from config, with fallback defaults for robustness.
MOUSE_SENSITIVITY = config.getfloat('Sensitivity', 'mouse', fallback=1.2)
SCROLL_SENSITIVITY = config.getfloat('Sensitivity', 'scroll', fallback=250.0)
CLICK_THRESHOLD = config.getfloat('Clicking', 'threshold', fallback=0.045)
ALPHA = config.getfloat('Smoothing', 'alpha', fallback=0.25)
SHOW_LANDMARKS = config.getboolean('Display', 'show_landmarks', fallback=True)

# Validate mouse sensitivity to ensure it's a valid value for our logic.
if MOUSE_SENSITIVITY < 1.0:
    print("Mouse sensitivity must be >= 1.0! Check config.ini.")
    sys.exit(0)

# Screen dimensions for scaling pointer coordinates.
width, height = pyautogui.size()

# Global variables to manage application state.
smooth_x, smooth_y = 0, 0
host_os = sys.platform
pt = time.perf_counter()  # Performance timer for calculating delta time.
state = [0, 0, 0, 0, 0]   # Represents the state of 5 fingers (1=open, 0=closed).
previous_state = state.copy()
handedness = None
mouse_down = False
scrolling = False
scroll_y = 0

# A dictionary to store kinematic data for each of the 21 hand landmarks.
Lms = {i: {"px": 0, "py": 0, "vx": 0, "vy": 0, "nx": 0, "ny": 0} for i in range(21)}


# --- 3. CORE ACTION FUNCTIONS ---

def move_pointer():
    """
    Moves the mouse pointer based on the index finger's position.
    This function uses an "active area box" for sensitivity control, which is the
    correct way to handle absolute position mapping.
    """
    global smooth_x, smooth_y
    if mouse_down or sum(state) == 5:
        # Define an active box. A higher sensitivity creates a smaller box,
        # meaning less hand movement is needed to cover the entire screen.
        box_size = 1.0 / MOUSE_SENSITIVITY
        offset = (1.0 - box_size) / 2.0
        
        raw_x, raw_y = Lms[8]['px'], Lms[8]['py']

        # Map the hand's position from within the active box to the full screen.
        screen_x_norm = np.interp(raw_x, [offset, offset + box_size], [0.0, 1.0])
        screen_y_norm = np.interp(raw_y, [offset, offset + box_size], [0.0, 1.0])

        # Clamp values to prevent the cursor from leaving the screen.
        screen_x_norm = np.clip(screen_x_norm, 0.0, 1.0)
        screen_y_norm = np.clip(screen_y_norm, 0.0, 1.0)

        # Scale normalized coordinates to actual screen pixels.
        x = width * screen_x_norm
        y = height * screen_y_norm
        
        # Apply exponential moving average for smoothing to reduce jitter.
        smooth_x = ALPHA * x + (1 - ALPHA) * smooth_x
        smooth_y = ALPHA * y + (1 - ALPHA) * smooth_y
        
        pyautogui.moveTo(smooth_x, smooth_y)


def right_click():
    """Performs a right-click. Triggered by a closed fist gesture."""
    pyautogui.rightClick()


def left_press():
    """
    Performs a left-click (mouse down/up). Triggered by pinching the
    thumb and index finger.
    """
    global mouse_down
    # Calculate the distance between the thumb tip (4) and index finger tip (8).
    dist = math.sqrt((Lms[4]['nx'] - Lms[8]['nx'])**2 + (Lms[4]['ny'] - Lms[8]['ny'])**2)
    
    if dist < CLICK_THRESHOLD:
        if not mouse_down:  # Press the mouse down only on the first frame of the pinch.
            pyautogui.mouseDown()
            mouse_down = True
    elif mouse_down:  # If fingers are apart and mouse was down, release it.
        pyautogui.mouseUp()
        mouse_down = False


def minimize():
    """Minimizes the current window."""
    if host_os.startswith('win'):
        pyautogui.hotkey('win', 'down')
    elif host_os == "darwin":
        pyautogui.hotkey('command', 'm')


def maximize():
    """Maximizes the current window."""
    if host_os.startswith('win'):
        pyautogui.hotkey('win', 'up')
    elif host_os == "darwin":
        pyautogui.hotkey('ctrl', 'command', 'f')
    elif host_os.startswith('linux'):
        pyautogui.hotkey('super', 'up')


def scroll():
    """Scrolls vertically based on the hand's vertical movement."""
    global scroll_y, scrolling
    # On the first frame of the scroll gesture, record the starting hand position.
    if not scrolling:
        scrolling = True
        scroll_y = Lms[8]['py']
    
    # Calculate the vertical distance moved from the starting point.
    dy = scroll_y - Lms[8]['py']
    scroll_amt = int(dy * SCROLL_SENSITIVITY)
    pyautogui.scroll(scroll_amt)


# --- 4. HAND AND GESTURE PROCESSING FUNCTIONS ---

def set_state():
    """Determines the binary state (open/closed) of each of the five fingers."""
    global handedness
    # Thumb is checked horizontally relative to its knuckle.
    if (handedness == "Right" and Lms[4]['nx'] < Lms[3]['nx']) or \
       (handedness == "Left" and Lms[4]['nx'] > Lms[3]['nx']):
        state[0] = 1
    else:
        state[0] = 0

    # Other four fingers are checked vertically.
    # A finger is "open" if its tip is above its middle knuckle.
    if Lms[8]['ny'] < Lms[6]['ny']: state[1] = 1
    else: state[1] = 0
    if Lms[12]['ny'] < Lms[10]['ny']: state[2] = 1
    else: state[2] = 0
    if Lms[16]['ny'] < Lms[14]['ny']: state[3] = 1
    else: state[3] = 0
    if Lms[20]['ny'] < Lms[18]['ny']: state[4] = 1
    else: state[4] = 0


def rotate():
    """
    Rotates all hand landmarks to a standard "upright" orientation.
    This makes gesture detection rotation-invariant and more reliable.
    """
    global Lms
    # Use the wrist (0) as the pivot and the middle finger knuckle (9) for orientation.
    pivot_x, pivot_y = Lms[0]['px'], Lms[0]['py']
    p9_x, p9_y = Lms[9]['px'], Lms[9]['py']
    
    current_angle = math.atan2(p9_y - pivot_y, p9_x - pivot_x)
    target_angle = -math.pi / 2  # Target is straight up.
    rotation_angle = target_angle - current_angle
    
    cos_rad = math.cos(rotation_angle)
    sin_rad = math.sin(rotation_angle)

    for i in range(21):
        translated_x = Lms[i]['px'] - pivot_x
        translated_y = Lms[i]['py'] - pivot_y
        Lms[i]['nx'] = translated_x * cos_rad - translated_y * sin_rad
        Lms[i]['ny'] = translated_x * sin_rad + translated_y * cos_rad


def get_landmarks(hand_landmarks):
    """Updates the position and calculates the velocity of each landmark."""
    global pt
    t = time.perf_counter()
    dt = t - pt
    pt = t
    for id, lm in enumerate(hand_landmarks.landmark):
        x, y = lm.x, lm.y
        dx = x - Lms[id]['px']
        dy = y - Lms[id]['py']
        if dt > 0:  # Avoid division by zero on the first frame.
            Lms[id]["vx"] = dx / dt
            Lms[id]["vy"] = dy / dt
        Lms[id]["px"] = x
        Lms[id]["py"] = y
    rotate()


# --- 5. MAIN APPLICATION LOOP ---

while capture.isOpened():
    # 1. Read a frame from the camera.
    success, frame = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 2. Pre-process the frame for MediaPipe.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 3. If a hand is detected, process gestures.
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Optionally draw the landmarks for visual feedback.
            if SHOW_LANDMARKS:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Update landmark data and determine the current gesture.
            get_landmarks(hand_landmarks)
            hand_handedness = results.multi_handedness[idx]
            handedness = hand_handedness.classification[0].label
            set_state()
            
            # --- Continuous Actions (run every frame the gesture is active) ---
            move_pointer()
            
            # The click gesture is very broad, which can lead to accidental
            # clicks when moving the pointer or scrolling.
            if sum(state) >= 4:
                left_press()

            if state == [0, 1, 1, 1, 1]:
                scroll()
            else:
                scrolling = False
            
            # --- One-Shot Actions (run only ONCE when the gesture changes) ---
            if state != previous_state:
                if state == [0, 1, 1, 0, 0]:
                    minimize()
                elif not sum(state):
                    right_click()
                elif state == [0, 1, 0, 0, 1]:
                    maximize()
                previous_state = state.copy()
    
    # 4. Display the processed frame.
    cv2.imshow('Pointer Control', frame)

    # 5. Handle exit conditions (press 'q' or make the exit gesture).
    if cv2.waitKey(1) & 0xFF == ord('q') or state == [0, 0, 1, 0, 0]:
        break

# --- 6. CLEANUP ---
pyautogui.mouseUp()  # Ensure the mouse is not left in a clicked state.
capture.release()
cv2.destroyAllWindows()