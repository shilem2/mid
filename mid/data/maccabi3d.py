import pandas as pd

from mid.data import Dataset
from .read_annotations import get_scan_anns, filter_anns_df
from mid.config import config


class Maccbi3dDataset(Dataset):
    """Dataset class, enables to get annotations, filter data, etc.
    """

    def __init__(self,
                 ):

        pass