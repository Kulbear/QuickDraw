import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import model_zoo

from torchvision import models

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import cv2
import ast
import time
import math
import pickle
import random
import datetime as dt
from collections import defaultdict, OrderedDict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
