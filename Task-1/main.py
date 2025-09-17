import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image
import time
import math

# --- Setup for MediaPipe and OpenCV ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence = 0.7, model_complexity = 0)
mp_draw = mp.solutions.drawing_utils
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Global Timestamps and State Variables ---
pt = time.perf_counter()  # Performance timer for calculating frame delta time (dt).
pst = time.perf_counter() # Timestamp for swipe cooldown.

# Dictionary to store kinematics (position, velocity) for each of the 21 landmarks.
Lms = {}
for i in range(21):
    Lms[i] = {
        "px": 0, "py": 0, "vx": 0, "vy": 0, "nx": 0, "ny": 0
    }

# A 5-element list representing the binary state (1=open, 0=closed) of the five fingers.
state = [0, 0, 0, 0, 0]
previous_state = [0, 0, 0, 0, 0]
# A sequence of 'state' lists that form the unlock pattern.
password = [[1, 1, 1, 1, 1]]

# User-defined gestures and their corresponding names and timers.
gestures = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]
names = ["Open Palm/Unlock", "Closed Fist/Lock", "Middle Finger/Exit"]
last_detected_times = [0, 0, 0]
gesture_cooldown = 1

# UI and application state variables.
menu = 0
layer = 0
locked = False
handedness = None
is_running = True
is_tracking_swipe = False
swipe_start_x = 0
swipe_start_time = 0

# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("Gesture Control")
root.configure(bg="#333333") # Darker background for better contrast

# Apply a modern theme
style = ttk.Style(root)
style.theme_use("clam")

# Configure styles for widgets for a consistent look
style.configure("TLabel", 
                background="#333333", 
                foreground="white", 
                font=("Segoe UI", 16, "bold"))
style.configure("TButton", 
                font=("Segoe UI", 12, "bold"), 
                padding=10)
style.configure("Header.TLabel", 
                font=("Segoe UI", 20, "bold", "italic"))

screen = ttk.Label(root)
screen.grid(row=1, column=0, columnspan=10, rowspan=10, padx=10, pady=10)


def exit_button_pressed():
    """Sets the flag to gracefully shut down the application loop."""
    global is_running
    is_running = False
    print(f'{time.perf_counter()}: Exit')


def swiped_left():
    """Navigates to the menu on the left."""
    global menu
    menu -= 1
    print(f'{time.perf_counter()}: Swiped Left')
    menu_load()


def swiped_right():
    """Navigates to the menu on the right."""
    global menu
    menu += 1
    print(f'{time.perf_counter()}: Swiped Right')
    menu_load()


def set_state():
    """Determines which fingers are open or closed based on landmark positions."""
    global handedness
    # The thumb is checked horizontally based on its x-coordinate.
    if (handedness == "Right" and Lms[4]['nx'] < Lms[3]['nx']) or \
       (handedness == "Left" and Lms[4]['nx'] > Lms[3]['nx']):
        state[0] = 1
    else:
        state[0] = 0

    # For other fingers, check if the tip's y-coordinate is above its knuckle's y-coordinate.
    if Lms[8]['ny'] < Lms[6]['ny']: state[1] = 1
    else: state[1] = 0
    if Lms[12]['ny'] < Lms[10]['ny']: state[2] = 1
    else: state[2] = 0
    if Lms[16]['ny'] < Lms[14]['ny']: state[3] = 1
    else: state[3] = 0
    if Lms[20]['ny'] < Lms[18]['ny']: state[4] = 1
    else: state[4] = 0


def rotate():
    """Normalizes hand landmark coordinates to make them rotation-invariant."""
    global Lms
    # Use the wrist (landmark 0) as the pivot point.
    pivot_x, pivot_y = Lms[0]['px'], Lms[0]['py']
    p9_x, p9_y = Lms[9]['px'], Lms[9]['py']
    
    # Calculate the hand's current angle from the vector between the wrist and middle finger base.
    current_angle = math.atan2(p9_y - pivot_y, p9_x - pivot_x)
    target_angle = -math.pi / 2  # Target angle is straight up (-90 degrees).
    rotation_angle = target_angle - current_angle
    
    # Apply the 2D rotation matrix formula to each landmark's coordinates.
    cos_rad = math.cos(rotation_angle)
    sin_rad = math.sin(rotation_angle)
    for i in range(21):
        translated_x = Lms[i]['px'] - pivot_x
        translated_y = Lms[i]['py'] - pivot_y
        Lms[i]['nx'] = translated_x * cos_rad - translated_y * sin_rad
        Lms[i]['ny'] = translated_x * sin_rad + translated_y * cos_rad


def get_landmarks(hand_landmarks):
    """Calculates and stores the position and velocity of all 21 hand landmarks."""
    global pt
    t = time.perf_counter()
    # Calculate delta time (dt) for frame-rate independent velocity.
    dt = t - pt
    pt = t
    for id, lm in enumerate(hand_landmarks.landmark):
        x = lm.x
        y = lm.y
        dx = x - Lms[id]['px']
        dy = y - Lms[id]['py']
        # Velocity = Change in Position / Change in Time.
        if dt > 0:
            Lms[id]["vx"] = dx / dt
            Lms[id]["vy"] = dy / dt
        Lms[id]["px"] = x
        Lms[id]["py"] = y
    rotate()


def locked_func():
    """Placeholder function assigned to buttons when the UI is locked."""
    pass


def clear_pass():
    """Resets the unlock password to the default gesture."""
    global password, layer
    password = [[1, 1, 1, 1, 1]]
    layer = 0


def append_pass():
    """Adds the current hand gesture to the unlock password sequence."""
    global password, state
    password.append(state.copy())


def create_gesture():
    """Saves the current gesture with a user-provided name."""
    global gestures, names, last_detected_times
    current = state.copy()
    entry_name = name.get()
    if current not in gestures:
        if entry_name:
            if entry_name in names:
                messagebox.showerror("Error", message="Name is already used")
            else:
                gestures.append(current)
                names.append(entry_name)
                last_detected_times.append(0)
        else:
            messagebox.showerror("Error", message="Input a name for the gesture.")
    else:
        messagebox.showerror("Error", message="Gesture already mapped")


def reset_gesture():
    """Resets the list of custom gestures to the default ones."""
    global gestures, names, last_detected_times
    gestures = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    names = ["open palm", "closed fist"]
    last_detected_times = [0, 0]


def to_english(st):
    """Helper function to convert binary finger state (0/1) to a readable string."""
    if st == 0:
        return "Closed"
    elif st == 1:
        return "Open"


def list_gestures():
    """Displays a message box listing all currently defined gestures."""
    info = "Swipe Left\n\nSwipe Right\n\n"
    for i in range(len(gestures)):
        info += f'{names[i]}:\n'
        info += f'\tThumb: {to_english(gestures[i][0])}\n'
        info += f'\tIndex: {to_english(gestures[i][1])}\n'
        info += f'\tMiddle: {to_english(gestures[i][2])}\n'
        info += f'\tRing: {to_english(gestures[i][3])}\n'
        info += f'\tPinky: {to_english(gestures[i][4])}\n\n'
    messagebox.showinfo("Gesture List", message=info)


# --- Tkinter Widget Definitions ---
name = ttk.Entry(root, width=15)
exit_button = ttk.Button(root, text="EXIT", command=exit_button_pressed)
lock = ttk.Label(root, text="UNLOCKED")
swipe_left = ttk.Button(root, text="←", command=swiped_left)
swipe_right = ttk.Button(root, text="→", command=swiped_right)
clear = ttk.Button(root, text="CLEAR", command=clear_pass)
add = ttk.Button(root, text="ADD", command=append_pass)
create = ttk.Button(root, text="CREATE", command=create_gesture)
reset = ttk.Button(root, text="RESET", command=reset_gesture)
list_gesture = ttk.Button(root, text="LIST", command=list_gestures)


def menu_0():
    """Configures the UI for the main (home) menu."""
    clear.grid_forget()
    add.grid_forget()
    create.grid_forget()
    reset.grid_forget()
    name.grid_forget()
    list_gesture.grid_forget()
    exit_button.grid(row=11, column=2, columnspan=5, pady=10)
    lock.grid(row=0, column=2, columnspan=6, pady=10)
    swipe_left.grid(row=0, column=0, pady=10)
    swipe_right.grid(row=0, column=9, pady=10)


def menu_1():
    """Configures the UI for the password settings menu."""
    exit_button.grid_forget()
    swipe_right.grid_forget()
    lock.grid_configure(row=0, column=1, columnspan=6, pady=10)
    clear.grid(row=11, column=0, columnspan=3, pady=10)
    add.grid(row=11, column=5, columnspan=3, pady=10)


def menu_m1():
    """Configures the UI for the custom gesture creation menu."""
    exit_button.grid_forget()
    swipe_left.grid_forget()
    lock.grid_configure(row=0, column=3, columnspan=6, pady=10)
    create.grid(row=11, column=1, columnspan=2, pady=10)
    reset.grid(row=11, column=4, columnspan=2, pady=10)
    list_gesture.grid(row=11, column=7, columnspan=2, pady=10)
    name.grid(row=11, column=9, columnspan=1, pady=10)


def menu_load():
    """Loads the correct menu UI based on the global 'menu' variable."""
    global menu
    if menu > 1: menu = 1
    elif menu < -1: menu = -1
    if menu == 0: menu_0()
    elif menu == 1: menu_1()
    elif menu == -1: menu_m1()


def lock_gesture():
    """Handles the logic for locking and unlocking the UI via gesture sequences."""
    global password, layer, state, locked
    if not locked:
        # A closed fist gesture locks the system.
        if state == [0, 0, 0, 0, 0]:
            lock.config(text="LOCKED")
            exit_button.config(text="EXIT (🔒)", command=locked_func)
            swipe_left.config(text="← (🔒)", command=locked_func)
            swipe_right.config(text="→ (🔒)", command=locked_func)
            clear.config(text="CLEAR (🔒)", command=locked_func)
            add.config(text="ADD (🔒)", command=locked_func)
            create.config(text="CREATE (🔒)", command=locked_func)
            reset.config(text="RESET (🔒)", command=locked_func)
            list_gesture.config(text="LIST (🔒)", command=locked_func)
            layer = 0
            locked = True
            return

    if locked:
        # Check if the current gesture matches the required gesture at the current 'layer' of the password.
        if state == password[layer]:
            layer += 1
            lock.config(text=f'{layer}/{len(password)}')
            # If all layers of the password are correct, unlock the system.
            if layer == len(password):
                lock.config(text="UNLOCKED")
                exit_button.config(text="EXIT", command=exit_button_pressed)
                swipe_left.config(text="←", command=swiped_left)
                swipe_right.config(text="→", command=swiped_right)
                clear.config(text="CLEAR", command=clear_pass)
                add.config(text="ADD", command=append_pass)
                create.config(text="CREATE", command=create_gesture)
                reset.config(text="RESET", command=reset_gesture)
                list_gesture.config(text="LIST", command=list_gestures)
                locked = False
        # If the wrong gesture is shown (and it's not a fist, which is neutral), reset progress.
        elif state != [0, 0, 0, 0, 0]:
            layer = 0
            lock.config(text=f'WRONG: 0/{len(password)}')


def exit_gesture():
    """Checks for the 'middle finger' gesture to close the application."""
    if state == [0, 0, 1, 0, 0]:
        exit_button.invoke()


def swipe():
    """Detects a simple swipe gesture based on instantaneous velocity and hand pose."""
    global pst, state
    # Checks if the exit button has been pressed, the purpose of this so there is no data race.
    if not is_running:
        capture.release()
        root.destroy()
        return
    # Only detect swipes if the hand is in an open palm pose to prevent accidental triggers.
    if sum(state) >= 3:
        velocity_threshold = 1.5
        cooldown = 0.75
        if Lms[5]['px'] == 0 or Lms[17]['px'] == 0:
            return
        
        # Calculate the average velocity of the palm's width (index to pinky base).
        center_vx = (Lms[5]['vx'] + Lms[17]['vx']) / 2

        # Check if the velocity exceeds the threshold and the cooldown has passed.
        if center_vx > velocity_threshold:
            if time.perf_counter() - pst > cooldown:
                swipe_right.invoke()
                pst = time.perf_counter()
        elif center_vx < -velocity_threshold:
            if time.perf_counter() - pst > cooldown:
                swipe_left.invoke()
                pst = time.perf_counter()


def check_custom_gestures():
    """Checks the current hand pose against the user-defined gesture list."""
    global last_detected_times
    for i, gesture in enumerate(gestures):
        if state == gesture:
            # If a match is found, check its personal cooldown timer to prevent spamming.
            if time.perf_counter() - last_detected_times[i] > gesture_cooldown:
                gesture_name = names[i]
                print(f"{time.perf_counter():.2f}: {gesture_name}")
                last_detected_times[i] = time.perf_counter()
                break


def show_frame():
    """The main application loop; processes each frame from the webcam."""
    global previous_state, handedness, is_running
    # Graceful shutdown check.
    if not is_running:
        capture.release()
        root.destroy()
        return

    ret, frame = capture.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # The order of these calls is critical for correct gesture detection.
                get_landmarks(hand_landmarks)
                hand_handedness = results.multi_handedness[idx]
                handedness = hand_handedness.classification[0].label
                set_state()

                # Only trigger state-based actions when the gesture changes.
                if state != previous_state:
                    lock_gesture()
                    check_custom_gestures()
                    previous_state = state.copy()

                # Trigger continuous actions like exit and swipe every frame.
                exit_gesture()
                swipe()
        
        # Convert the OpenCV image to a Tkinter-compatible format and display it.
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        screen.imgtk = imgtk
        screen.configure(image=imgtk)

    # Schedule this function to run again after 2ms, creating the video loop.
    root.after(2, show_frame)


# --- Start the Application ---
menu_load()
show_frame()
root.mainloop()