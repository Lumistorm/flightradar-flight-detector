import webbrowser
import pyautogui
import time
from mss import mss
import numpy as np
from ultralytics import YOLO
from random import randint, uniform
from PIL import Image, ImageDraw
import os
import pytesseract
import cProfile
import pstats
import time
import easyocr

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


DISPLAY_SIZE = pyautogui.size()
DISPLAY_WIDTH = DISPLAY_SIZE.width
DISPLAY_HEIGHT = DISPLAY_SIZE.height

MAP_WIDTH = ((DISPLAY_WIDTH * 295) // 378) - 1
MAP_HEIGHT = ((DISPLAY_HEIGHT * 861) // 982) - 1
pyautogui.PAUSE = 0.0


def init_website():
    url = 'https://www.flightradar24.com/45.24,-74.18/10'
    webbrowser.open(url)
    time.sleep(3)


def detect_aircraft(model, source, target_label):
    # predict aircraft locations
    start_time = time.perf_counter()
    results = model.predict(
        source=source,
        conf=0.25,
        device="mps",
        imgsz=640,
        verbose=False,
        save=False
    )
    print('yolo: ', time.perf_counter() - start_time)
    r = results[0]

    # source dimensions
    source_height, source_width = r.orig_shape

    # predict text locations
    start_time = time.perf_counter()
    text_data = pytesseract.image_to_data(source, output_type=pytesseract.Output.DICT)
    print('tesseract: ', time.perf_counter() - start_time)

    # get label normalized position and dimensions
    for index in range(len(text_data['text'])):
        text = text_data['text'][index].strip()
        print(text)

        if text.upper() == target_label.upper():
            text_x = text_data['left'][index]
            text_y = text_data['top'][index]

            break
    else:
        return

    # set to large number
    closest_distance = 1_000_000_000
    closest_center_x = 0
    closest_center_y = 0

    # find the closest aircraft to target label
    for x_center, y_center, width, height in r.boxes.xywh:
        # get middle left position
        x = x_center + width * 0.5

        # get distance to target label
        dx = float(x - text_x)
        dy = float(y_center - text_y)
        distance_squared = dx * dx + dy * dy
        if distance_squared < closest_distance:
            closest_distance = distance_squared
            closest_center_x = x_center
            closest_center_y = y_center
    print(closest_center_x, closest_center_y)

    pyautogui.moveTo(closest_center_x / source_width * DISPLAY_WIDTH, closest_center_y / source_height * DISPLAY_HEIGHT)
    pyautogui.mouseDown(button='left')
    pyautogui.mouseUp(button='left')
    # time.sleep(2)
    # pyautogui.moveTo(336, 155)
    # pyautogui.mouseDown(button='left')
    # pyautogui.mouseUp(button='left')
    # time.sleep(1)


def scan_map(x_range, y_range, model, target_label):
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
                time.sleep(0.1)
                pyautogui.mouseUp(button='left')
                time.sleep(0.2)

        if y < y_range - 1:
            # reset mouse position
            pyautogui.moveTo(MAP_WIDTH // 2, DISPLAY_HEIGHT - 1)

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


def generate_augmented_batch(batch_size):
    # define paths
    inputs_path = 'dataset/val/images'
    debug_path = 'dataset/val/debug_inputs'
    labels_path = 'dataset/val/labels'
    os.makedirs(inputs_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # load images
    backgrounds = [
        Image.open(f'map_backgrounds/map_background_{i}.png').convert('RGBA')
        for i in range(200)
    ]

    aircraft = [
        Image.open(f'yellow_aircraft/aircraft_{i}.png').convert('RGBA')
        for i in range(31)
    ]

    count = 1
    for index in range(batch_size):
        labels = []

        background_img = backgrounds[randint(0, 199)].copy()
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
                aircraft_img = aircraft[randint(0, 30)].copy()
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

                background_width, background_height = background_img.size
                aircraft_width, aircraft_height = aircraft_img.size

                # offscreen threshold
                max_offscreen_x = aircraft_width // 4
                max_offscreen_y = aircraft_height // 4

                random_x = randint(-max_offscreen_x, 2360 - max_offscreen_x)
                random_y = randint(-max_offscreen_y, background_height - max_offscreen_y)

                background_img.paste(aircraft_img, (random_x, random_y), aircraft_img)
                debug_img.paste(aircraft_img, (random_x, random_y), aircraft_img)

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

                # draw bounding box
                box_corners = [box_x1, box_y1, box_x2, box_y2]
                draw.rectangle(box_corners, outline='red', width=2)

        # save
        background_img = background_img.convert('RGB')
        debug_img = debug_img.convert('RGB')

        background_img.save(f'{inputs_path}/image_{index}.png')
        debug_img.save(f'{debug_path}/image_{index}.png')

        with open(f'{labels_path}/image_{index}.txt', 'w') as f:
            f.write('\n'.join(labels))

        print(f'Done {count} / {batch_size}')
        count += 1


def train_model():
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


def main():
    # train_model()

    # generate_augmented_batch(400)
    # model = init_YOLO()
    #

    # while True:
    #     print(pyautogui.position())
    #     time.sleep(3)
    init_website()

    # init models
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    model = YOLO('runs/train/weights/best.mlmodel')
    model.predict(dummy)

    pytesseract.image_to_data(dummy)

    scan_map(1, 1, model, 'ACA808')


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    results = pstats.Stats(profiler).sort_stats('cumtime')
    results.print_stats(20)

