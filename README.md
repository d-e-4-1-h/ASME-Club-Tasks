# ASME-Club-Tasks

# For Task-1

# Real-Time Hand Gesture Control UI

This Python application leverages your webcam to provide real-time control over a graphical user interface using hand gestures. It is built with OpenCV for video capture, MediaPipe for advanced hand tracking, and Tkinter for the UI.

---

## ✨ Key Features

-   **Real-time Hand Tracking:** Utilizes Google's MediaPipe to accurately detect 21 hand landmarks.
-   **Gesture-Based UI Control:** Navigate menus, lock/unlock the interface, and trigger actions using intuitive hand poses.
-   **Swipe Navigation:** Quickly move your hand left or right to switch between different UI menus.
-   **Customizable Security:**
    -   Lock the UI with a **closed fist**.
    -   Set a custom multi-gesture sequence as your password to unlock.
-   **Custom Gesture Creator:** Define, name, and save your own unique hand gestures to be recognized by the application.
-   **Rotation Invariance:** Gesture detection remains robust even when your hand is tilted, thanks to coordinate normalization.
-   **Clean UI:** A simple and modern interface built with Tkinter.

---

## 🔧 Requirements

Make sure you have Python 3 installed. You will need the following libraries:

-   `opencv-python`
-   `mediapipe`
-   `Pillow` (PIL)

### ⚙️ Installation

1.  **Clone the repository:**
    ```sh
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install the required packages using pip:**
    ```sh
    pip install opencv-python mediapipe Pillow
    ```

---

## ▶️ How to Run

Simply execute the `main.py` script from your terminal:


# For Task-2

# Gesture Control for Your Computer


This Python script transforms your webcam into a powerful input device, allowing you to control your computer's mouse, clicks, scrolling, and window management using intuitive hand gestures. It's built with **OpenCV** for video processing and **Google's MediaPipe** for robust, real-time hand tracking.

The entire system is designed to be highly configurable through a simple `config.ini` file, so you can easily tune sensitivity and other parameters to your liking without touching the code.

---

## ✨ Features

-   **Mouse Pointer Control**: Move your cursor by moving your index finger.
-   **Clicking & Dragging**: Pinch your thumb and index finger together to perform a left-click. Hold the pinch to drag and drop.
-   **Right-Click**: Make a closed fist to perform a right-click.
-   **Scrolling**: Use a "four fingers up" gesture to enter scroll mode. Move your hand up and down to scroll.
-   **Window Management**:
    -   **Minimize**: Show your index and middle fingers to minimize the active window.
    -   **Maximize**: Make a "rock on" gesture (index and pinky up) to maximize the window.
-   **High Performance**: Optimized for speed with a lightweight hand tracking model and configurable resolution.
-   **Customizable Settings**: Easily adjust mouse sensitivity, scrolling speed, click distance, and smoothing via the `config.ini` file.
-   **Cross-Platform**: Works on Windows, macOS, and Linux.

---

## 🔧 Setup and Installation

### Prerequisites

-   Python 3.6 or newer
-   A webcam

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install Required Libraries**
    Install all necessary Python packages using pip.
    ```bash
    pip install opencv-python mediapipe pyautogui numpy
    ```

3.  **Create a Configuration File**
    Create a file named `config.ini` in the same directory as the script. Copy and paste the following content into it. You can adjust these values later to fine-tune performance.

    ```ini
    [Sensitivity]
    # Higher value means a smaller hand movement covers the whole screen. Must be >= 1.0
    mouse = 1.2
    # Controls how fast the page scrolls.
    scroll = 250.0

    [Clicking]
    # The distance between thumb and index finger to trigger a click (smaller is more sensitive).
    threshold = 0.045

    [Smoothing]
    # Controls mouse pointer smoothing. Value between 0.0 and 1.0.
    # Lower values = smoother but more lag. Higher values = more responsive but jittery.
    alpha = 0.25

    [Display]
    # Set to true to see the hand landmarks drawn on the camera feed for debugging.
    show_landmarks = true
    ```

---

## ▶️ How to Run

Simply execute the `main.py` script from your terminal:
