from quick_draw.common_import import *
from quick_draw.configs import *
from quick_draw.utils import get_country_information

try:
    with open(COUNTRY_INFO_FILE, 'rb') as handle:
        country_mapping = pickle.load(handle)
except FileNotFoundError:
    get_country_information()
except Exception:
    raise NotImplementedError('No country information found, cannot proceed!')
finally:
    with open(COUNTRY_INFO_FILE, 'rb') as handle:
        country_mapping = pickle.load(handle)


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


def drop_draw_cv2(raw_strokes, img_size=256, lw=8):
    img = np.zeros((RAW_IMG_SIZE, RAW_IMG_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            if i == 0 or i == len(stroke[0]) - 2:
                color = 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if img_size != RAW_IMG_SIZE:
        return cv2.resize(img, (img_size, img_size))
    else:
        return img


def build_image(df, img_size, lw=6, time_color=True):
    x = np.zeros((len(df), img_size, img_size, 3))
    for i, raw_strokes in enumerate(df.drawing.values):
        img = draw_cv2(raw_strokes, img_size=img_size, lw=lw,
                       time_color=time_color)
        mask = (img < 1).astype(np.uint8)
        c_value = country_mapping.get(df.iloc[i]['countrycode'], 40)
        x[i, :, :, 0] = img + mask * c_value
        x[i, :, :, 1] = drop_draw_cv2(raw_strokes, img_size=img_size)
        x[i, :, :, 2] = draw_cv2(raw_strokes, img_size=img_size, lw=lw + 3,
                                 time_color=False)

    return x


def image_generator_xd(img_size, batch_size, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            with open('train_file_log.txt', 'a+') as f:
                f.write('train_k{}.csv.\n'.format(k))
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = build_image(df, img_size, lw=lw, time_color=time_color)
                x = x / 255.
                y = np.array(list(df.y))
                yield x, y


def val_image_generator_xd(img_size, lw=6, time_color=True, batch_size=100):
    filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(99))
    counter = 0
    factor = 2
    for df in pd.read_csv(filename, chunksize=batch_size * factor):
        if counter >= NUM_CLASS // factor:
            return None
        df['drawing'] = df['drawing'].apply(ast.literal_eval)
        x = build_image(df, img_size, lw=lw, time_color=time_color)
        x = x / 255.
        y = np.array(list(df.y))
        counter += 1
        yield x, y


def test_image_generator_xd(img_size, lw=6, time_color=True, batch_size=500):
    filename = os.path.join(INPUT_DIR, 'test_simplified.csv')
    for df in tqdm(pd.read_csv(filename, chunksize=batch_size)):
        df['drawing'] = df['drawing'].apply(ast.literal_eval)
        x = build_image(df, img_size, lw=lw, time_color=time_color)
        x = x / 255.
        yield x


def df_to_image_array_xd(df, img_size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = build_image(df, img_size, lw=lw, time_color=time_color)
    x = x / 255.
    return x
