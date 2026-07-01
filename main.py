import webbrowser
import pyautogui
import time
from mss import mss
import numpy as np
from ultralytics import YOLO


DISPLAY_SIZE = pyautogui.size()
MAP_WIDTH = ((DISPLAY_SIZE.width * 295) // 378) - 1
MAP_HEIGHT = ((DISPLAY_SIZE.height * 861) // 982) - 1
pyautogui.PAUSE = 0.0


def init_YOLO():
    model = YOLO("yolov8n.pt")

    return model


def init_website():
    url = 'https://www.flightradar24.com/45,-75/8'
    webbrowser.open(url)
    time.sleep(3)


def scan_map(x_range, y_range, model):
    direction = 1
    for y in range(y_range):
        for x in range(x_range):
            # reset mouse position
            if direction == 1:
                pyautogui.moveTo(MAP_WIDTH, DISPLAY_SIZE.height // 2)
            else:
                pyautogui.moveTo(1, DISPLAY_SIZE.height // 2)

            # take screenshot
            img_array = get_screenshot()

            # ########## process image ##########

            if x < x_range - 1:
                pyautogui.mouseDown(button='left')
                pyautogui.move(-direction * MAP_WIDTH, 0, duration=1.5)
                time.sleep(0.1)
                pyautogui.mouseUp(button='left')
                time.sleep(0.2)

        if y < y_range - 1:
            # reset mouse position
            pyautogui.moveTo(MAP_WIDTH // 2, DISPLAY_SIZE.height - 1)

            # drag down
            pyautogui.mouseDown(button='left')
            pyautogui.move(0, -MAP_HEIGHT, duration=1.5)
            time.sleep(0.1)
            pyautogui.mouseUp(button='left')
            time.sleep(0.2)

            direction *= -1


def get_screenshot():
    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        # convert image to BGR array
        img_array = np.array(screenshot)[..., :3]

    return img_array


def get_airplane_icons_positions(model, image):
    results = model(image, verbose=False, stream=True)
    

if __name__ == '__main__':
    model = init_YOLO()

    init_website()
    scan_map(2, 2, model)

