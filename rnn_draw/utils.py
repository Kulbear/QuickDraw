from rnn_draw.common_import import *
from rnn_draw.configs import *


def mkdir(dir_path):
    try:
        os.listdir(dir_path)
    except FileNotFoundError:
        os.mkdir(dir_path)


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def _stack_it(raw_strokes):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    stroke_vec = ast.literal_eval(raw_strokes)  # string->list
    # unwrap the list
    in_strokes = [(xi, yi, i)
                  for i, (x, y) in enumerate(stroke_vec)
                  for xi, yi in zip(x, y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:, 2] = [1] + np.diff(c_strokes[:, 2]).tolist()
    c_strokes[:, 2] += 1  # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1),
                         maxlen=STROKE_COUNT,
                         padding='post').swapaxes(0, 1)


def get_country_information():
    tables = pd.read_html(WIKI_URL)
    df = tables[2]
    df.columns = ['Continent', 'Country', 'a', 'b', 'c']
    df = df[['Continent', 'Country']]

    d = defaultdict(int)
    for idx in df.iterrows():
        if idx[0] == 0:
            continue
        row = idx[1]
        if row.Continent == 'NA':
            d[row.Country] = 0
        elif row.Continent == 'AS':
            d[row.Country] = 160
        elif row.Continent == 'EU':
            d[row.Country] = 80
        elif row.Continent == 'AF':
            d[row.Country] = 240
        elif row.Continent == 'OC':
            d[row.Country] = 80
        else:
            d[row.Country] = 0
    d.pop('a-2', None)

    with open(COUNTRY_INFO_FILE, 'wb') as f:
        pickle.dump(d, f)


def f2cat(filename: str) -> str:
    return filename.split('.')[0]


def list_all_categories() -> list:
    try:
        with open(CATEGORY_LIST, 'rb') as f:
            result = pickle.load(f)
    except FileNotFoundError:
        files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
        result = sorted([f2cat(f) for f in files], key=str.lower)
        with open(CATEGORY_LIST, 'wb') as f:
            pickle.dump(result, f)
    return result


def data_generator_xd(batch_size, ks):
    while True:
        for k in np.random.permutation(ks):
            with open('FullSetTrainLog.txt', 'a+') as f:
                f.write('complete_train_k{}.csv.\n'.format(k))
            filename = os.path.join(DP_DIR, 'complete_train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                df['drawing'] = df['drawing'].map(_stack_it)
                x = np.stack(df['drawing'], 0)
                y = np.array(list(df.y))
                yield x, y
        return None


def df_to_sequence_array(df):
    df['drawing'] = df['drawing'].map(_stack_it)
    x = np.stack(df['drawing'], 0)
    return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def init(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return str(self.avg)
