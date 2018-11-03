import os
import ast
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input

DP_DIR = './input/shuffle-csvs/'
INPUT_DIR = './input/quickdraw-doodle-recognition/'
RAW_IMG_SIZE = 256
NUM_CLASS = 340


def f2cat(filename: str) -> str:
    return filename.split('.')[0]


def list_all_categories() -> list:
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)


def draw_cv2(raw_strokes, img_size=256, lw=6, time_color=True):
    img = np.zeros((RAW_IMG_SIZE, RAW_IMG_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if img_size != RAW_IMG_SIZE:
        return cv2.resize(img, (img_size, img_size))
    else:
        return img


def image_generator_xd(img_size, batch_size, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), img_size, img_size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, img_size=img_size, lw=lw,
                                             time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NUM_CLASS)
                yield x, y


def df_to_image_array_xd(df, img_size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), img_size, img_size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, img_size=img_size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x
