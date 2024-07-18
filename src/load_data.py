import pandas as pd
from paths import param, rpt, diff, life, form, electrode
import itertools


def param_df():
    return pd.read_csv(param)

def rpt_df():
    return pd.read_csv(rpt)

def diff_df():
    return pd.read_csv(diff)

def life_df():
    return pd.read_csv(life)

def form_df():
    return pd.read_csv(form)

def electrode_df():
    return pd.read_csv(electrode)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return list(itertools.zip_longest(*args, fillvalue=fillvalue))

seq_nums = list(range(100, 231)) + list(range(269, 327))
not_right = [111, 132, 133, 153, 218, 239, 164]
batch_1 = [None if x in not_right else x for x in seq_nums]
repeats = grouper(batch_1, 3)