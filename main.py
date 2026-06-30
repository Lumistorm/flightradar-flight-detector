import webbrowser
import pyautogui
import time
from mss import mss
import cv2
import numpy as np
from ultralytics import YOLO


def init_YOLO():
    model = YOLO("yolo11n.pt")

    return model


def init_website():
    url = 'https://www.flightradar24.com/45.29,-76.12/8'
    webbrowser.open(url)
    time.sleep(3)

    pyautogui.screenshot()


def get_screenshot():
    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        screen_image = np.array(screenshot)
        screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)

        filename = "screenshot_test.png"
        cv2.imwrite(filename, screen_image)


def get_airplane_icons_positions(model, image):
    results = model(image, verbose=False, stream=True)
    


if __name__ == '__main__':
    init_website()
    get_screenshot()
