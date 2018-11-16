from quick_draw.common_import import *
from quick_draw.configs import *


def mkdir(dir_path):
    try:
        os.listdir(dir_path)
    except FileNotFoundError:
        os.mkdir(dir_path)


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
