
import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

def open_file(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')