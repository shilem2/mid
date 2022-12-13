import os
import pickle
import blosc
from mid.data.utils import load_compressed_pickle
from mid.data import Annotation

def read_test_data(file_name='MR04-019.dat'):

    data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    file_name_full = os.path.join(data_dir, file_name)
    with open(file_name_full, 'rb') as f:
        compressed = f.read()
    pickled = blosc.decompress(compressed)
    data = pickle.loads(pickled)
    ann1, ann2, img1, img2, pixel_spacing, units, transform = data

    # add missing attributes (added after saving test data)
    if isinstance(ann1, Annotation) and not hasattr(ann1, 'flipped_anns'):
        ann1.flipped_anns = None
    if isinstance(ann2, Annotation) and not hasattr(ann2, 'flipped_anns'):
        ann2.flipped_anns = None

    return ann1, ann2, img1, img2, pixel_spacing, units, transform


def read_data(file_name='MR04-001.dat'):

    data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    file_name_full = os.path.join(data_dir, file_name)
    data = load_compressed_pickle(file_name_full)
    ann1, ann2, transform = data
    ann1.dicom_path = os.path.join(data_dir, ann1.dicom_path)
    ann2.dicom_path = os.path.join(data_dir, ann2.dicom_path)

    return ann1, ann2, transform

