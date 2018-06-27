import collections
from itertools import compress
import numpy as np

t = [False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, False]
index = np.where(t)[0]

index_1 = list(compress(np.arange(len(t)), t))

