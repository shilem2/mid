import json
from pathlib import Path
from tqdm import tqdm

from mid.data import utils
from mid.data.read_3d_data import read_metadata_single_dir, read_metadata_root_dir


def read_metadata_single_dir_example():

    dir_path = 'm:/magic/output/Pre_post_CT_XR_cohort/20220328072248/1003813_1_2_9c4f7c2e-cdfd-457e-9e24-7895bde42c9c/'
    metadata = read_metadata_single_dir(dir_path)

    pass


def read_metadata_root_dir_example():

    root_dir = 'm:/magic/output/Pre_post_CT_XR_cohort/'
    metadata = read_metadata_root_dir(root_dir)

    metadata_file = 'm:/moshe/3d_prediction/results/3d_db/metadata.dat'
    utils.save_compressed_pickle(metadata, metadata_file)

    metadata_file = 'm:/moshe/3d_prediction/results/3d_db/metadata.dat'
    metadata2 = utils.load_compressed_pickle(metadata_file)

    pass




if __name__ == '__main__':

    # read_metadata_single_dir_example()
    read_metadata_root_dir_example()

    pass