
import pyautogui
import os
import numpy as np
import logging
import webbrowser
from mss import mss
from ultralytics import YOLO
from random import randint, uniform
from PIL import Image, ImageDraw
from pytesseract import image_to_data, Output
from cProfile import Profile
from pstats import Stats
from time import sleep
from numba import njit

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# -------------------------
# Dimensions and display
# -------------------------

DISPLAY_SIZE = pyautogui.size()
DISPLAY_WIDTH = DISPLAY_SIZE.width
DISPLAY_HEIGHT = DISPLAY_SIZE.height

MAP_WIDTH = ((DISPLAY_WIDTH * 295) // 378) - 1
MAP_HEIGHT = ((DISPLAY_HEIGHT * 861) // 982) - 1
pyautogui.PAUSE = 0.0

sct = mss()
monitor = sct.monitors[1]

# -------------------------
# Initialize Logger
# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s : %(message)s'
)

logger = logging.getLogger()


def init_website():
    """
    Opens Flightradar24 website on Chrome.

    :return: None.
    """

    url = 'https://www.flightradar24.com/45.24,-74.18/10'

    webbrowser.open(url)
    sleep(3)


@njit
def get_label_position(texts, lefts, tops, target: str):
    """
    return position of target label.

    :param data: Predictions from tesseract.
    :param target: Target label.
    :return: The x and y of the target label.
    """

    for index in range(len(texts)):
        text = texts[index].strip()

        if text.upper() == target.upper():
            text_x = lefts[index]
            text_y = tops[index]

            return text_x, text_y
    else:
        return None, None


# @njit
def get_closest_box(boxes, x, y):
    """
    Return the closest box to target.

    :param boxes: List of boxes from YOLO.
    :param x: Target x.
    :param y: Target y.
    :return: The x and y of the closest box.
    """

    closest_distance = 1e18
    closest_x = 0
    closest_y = 0

    for box in boxes:
        x_center, y_center, width, height = box

        # get middle right position
        right_x = x_center + width * 0.5

        # get distance to target label
        dx = right_x - x
        dy = y_center - y
        distance_squared = dx * dx + dy * dy

        if distance_squared < closest_distance:
            closest_distance = distance_squared
            closest_x = x_center
            closest_y = y_center
        print(x_center, y_center)

    return closest_x, closest_y


def detect_aircraft(model, source: np.ndarray, target_label: str) -> None:
    """
    Clicks on the target aircraft.

    :param model: YOLO model.
    :param source: Numpy array of an image.
    :param target_label: Label of target aircraft.
    :return: None.
    """

    # predict aircraft locations
    prediction = model.predict(
        source=source,
        conf=0.25,
        device="mps",
        imgsz=640,
        verbose=False,
        save=False
    )
    r = prediction[0]

    # source dimensions
    source_height, source_width = r.orig_shape

    # predict text locations
    text_data = image_to_data(source, output_type=Output.DICT)

    # get label normalized position and dimensions
    text_x, text_y = get_label_position(
        texts=text_data['text'],
        lefts=text_data['left'],
        tops=text_data['top'],
        target=target_label,
    )

    if text_x is None:
        logger.info('Aircraft not detected')
        return

    # set to large number
    xywh = r.boxes.xywh.cpu().numpy()
    closest_x, closest_y = get_closest_box(boxes=xywh, x=text_x, y=text_y)

    # click on aircraft
    pyautogui.moveTo(
        closest_x / source_width * DISPLAY_WIDTH,
        closest_y / source_height * DISPLAY_HEIGHT
    )
    print(closest_x / source_width * DISPLAY_WIDTH, closest_y / source_height * DISPLAY_HEIGHT)
    pyautogui.mouseDown(button='left')
    pyautogui.mouseUp(button='left')

    # close menu
    # sleep(2)
    # pyautogui.moveTo(336, 155)
    # pyautogui.mouseDown(button='left')
    # pyautogui.mouseUp(button='left')
    # sleep(1)


def scan_map(x_range: int, y_range: int, model, target_label: str) -> None:
    """
    Scan the map to detect aircraft.

    :param x_range: Range of x scans.
    :param y_range: Range of y scans.
    :param model: YOLO model.
    :param target_label: Label of target aircraft.
    :return: None.
    """

    direction = 1
    for y in range(y_range):
        for x in range(x_range):
            # reset mouse position
            if direction == 1:
                pyautogui.moveTo(MAP_WIDTH, DISPLAY_HEIGHT // 2)
            else:
                pyautogui.moveTo(1, DISPLAY_HEIGHT // 2)

            # take screenshot
            img_array = get_screenshot()

            # ########## process image ##########
            detect_aircraft(model=model, source=img_array, target_label=target_label)

            # drag map to next location
            if x < x_range - 1:
                pyautogui.mouseDown(button='left')
                pyautogui.move(-direction * MAP_WIDTH, 0, duration=1.5)
                sleep(0.1)
                pyautogui.mouseUp(button='left')
                sleep(0.2)

        if y < y_range - 1:
            # reset mouse position
            pyautogui.moveTo(MAP_WIDTH // 2, DISPLAY_HEIGHT - 1)

            # drag down
            pyautogui.mouseDown(button='left')
            pyautogui.move(0, -MAP_HEIGHT, duration=1.5)
            sleep(0.1)
            pyautogui.mouseUp(button='left')
            sleep(0.2)

            direction *= -1


def get_screenshot() -> np.ndarray:
    """
    Get screenshot of display as numpy.ndarray.

    :return: None.
    """

    screenshot = sct.grab(monitor)

    # convert image to BGR array
    img_array = np.array(screenshot)[..., :3]

    return img_array


def generate_augmented_images(
        amount: int, *, inputs_path: str, debug_path: str, labels_path: str, save_debug=True
) -> None:
    """
    Generate and save augmented image for model training.

    :param amount: amount of images to generate.
    :param inputs_path: Path to save input images.
    :param debug_path: Path to save debug images.
    :param labels_path: Path to save labels.
    :param save_debug: Whether to save debug images.
    :return: None.
    """

    # create paths
    os.makedirs(inputs_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # load images
    num_aircraft = 31
    num_backgrounds = 200

    backgrounds = [
        Image.open(f'map_backgrounds/map_background_{i}.png').convert('RGBA')
        for i in range(num_backgrounds)
    ]
    background_width, background_height = backgrounds[0].size

    aircraft = [
        Image.open(f'yellow_aircraft/aircraft_{i}.png').convert('RGBA')
        for i in range(num_aircraft)
    ]

    count = 1
    for index in range(amount):
        labels = []

        background_img = backgrounds[randint(0, num_backgrounds - 1)].copy()
        debug_img = background_img.copy()

        # 10 % of data without target
        if count % 10:
            resize_factor = uniform(0.6, 1)
            draw = ImageDraw.Draw(debug_img)

            # compute random num planes
            r = np.random.random()

            if r < 0.25:
                num_planes = np.random.poisson(0.5)
            elif r < 0.7:
                num_planes = np.random.poisson(3)
            else:
                num_planes = np.random.poisson(10)

            num_planes = np.clip(num_planes, 1, 15)

            # loop for random num planes
            for _ in range(num_planes):
                aircraft_img = aircraft[randint(0, num_aircraft - 1)].copy()
                aircraft_width, aircraft_height = aircraft_img.size

                # resize
                resize_width = round(resize_factor * aircraft_width)
                resize_height = round(resize_factor * aircraft_height)
                aircraft_img = aircraft_img.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)

                # rotate
                random_angle = randint(0, 360)
                aircraft_img = aircraft_img.rotate(random_angle, resample=Image.BICUBIC, expand=True)

                # get tight rect box
                box_x1, box_y1, box_x2, box_y2 = aircraft_img.getbbox()

                aircraft_width, aircraft_height = aircraft_img.size

                # offscreen threshold
                max_offscreen_x = aircraft_width // 4
                max_offscreen_y = aircraft_height // 4

                random_x = randint(-max_offscreen_x, 2360 - max_offscreen_x)
                random_y = randint(-max_offscreen_y, background_height - max_offscreen_y)

                background_img.paste(aircraft_img, (random_x, random_y), aircraft_img)

                # bounding box coordinates
                box_x1 += random_x
                box_y1 += random_y
                box_x2 += random_x
                box_y2 += random_y

                # clamp coordinates
                box_x1 = max(0, box_x1)
                box_y1 = max(0, box_y1)
                box_x2 = min(background_width, box_x2)
                box_y2 = min(background_height, box_y2)
                box_width = box_x2 - box_x1
                box_height = box_y2 - box_y1

                # compute label values
                x_center = (box_x1 + (box_width / 2)) / background_width
                y_center = (box_y1 + (box_height / 2)) / background_height
                yolo_width = box_width / background_width
                yolo_height = box_height / background_height
                labels.append(f"{0} {x_center:.6f} {y_center:.6f} {yolo_width:.6f} {yolo_height:.6f}")

                # draw debug
                if save_debug:
                    box_corners = [box_x1, box_y1, box_x2, box_y2]
                    draw.rectangle(box_corners, outline='red', width=2)
                    debug_img.paste(aircraft_img, (random_x, random_y), aircraft_img)

        # save images
        background_img = background_img.convert('RGB')
        background_img.save(f'{inputs_path}/image_{index}.png')

        with open(f'{labels_path}/image_{index}.txt', 'w') as f:
            f.write('\n'.join(labels))

        if save_debug:
            debug_img = debug_img.convert('RGB')
            debug_img.save(f'{debug_path}/image_{index}.png')

        logger.info(f'Done {count} / {amount}')
        count += 1


def train_model() -> None:
    """
    Train and save YOLO model.

    :return: None.
    """

    model = YOLO('yolov8n.pt')

    model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs",
        workers=2,
        device='mps',
        patience=15,
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
    )


def main() -> None:

    # -------------------------
    # Train Yolo Model
    # -------------------------

    do_generate_dataset = False
    do_train_model = False

    if do_generate_dataset:
        generate_augmented_images(
            400,
            inputs_path='dataset/val/images',
            debug_path='dataset/val/debug_inputs',
            labels_path='dataset/val/labels',
            save_debug=True,
        )
    if do_train_model:
        train_model()

    # -------------------------
    # Find aircraft
    # -------------------------
    init_website()

    # init models
    logger.info('Initializing YOLO and tesseract models')
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model = YOLO('runs/train/weights/best.mlmodel')
    model.predict(dummy)
    image_to_data(dummy)
    logger.info('Finished initializing models')

    logger.info('Running aircraft detection (type \'q\' to quit).')
    pyautogui.hotkey('command', 'tab')
    sleep(0.1)

    while True:
        target_label = input('Enter target label: ')
        if target_label.lower() in ('q', 'quit', ''):
            break

        pyautogui.hotkey('command', 'tab')
        scan_map(1, 1, model, target_label)


if __name__ == '__main__':
    # profile code
    profiler = Profile()
    profiler.enable()

    main()

    profiler.disable()

    results = Stats(profiler).sort_stats('cumtime')
    results.print_stats(20)
