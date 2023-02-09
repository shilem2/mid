import json
from pathlib import Path
from tqdm import tqdm

from mid.data import utils


def read_metadata_single_dir(dir_path, patient_file='Patient.json', study_file='Study.json', study_analysis_file='StudyAnalysis.json'):

    dir_path = Path(dir_path)

    patient = json.load(open(dir_path / patient_file))
    study = json.load(open(dir_path / study_file))
    study_analysis = json.load(open(dir_path / study_analysis_file))

    study_id = patient['patientId'],  # Maccabi study id

    metadata = {
        'study_id': study_id,
        'mongo_id': study['patientId'],  # Mongo DB id
        'series_date': study['seriesDate'],
        'dir_path': dir_path.absolute().as_posix(),
        # 'study_analysis': study_analysis,
    }

    return metadata, study_id

def read_metadata_root_dir(root_dir, pattern='**/Patient.json'):

    paths = Path(root_dir).rglob(pattern)
    # path_list = list(paths)

    metadata_dict = {}
    for path in tqdm(paths):

        metadata, study_id = read_metadata_single_dir(path.parent)

        if study_id not in metadata_dict:
            metadata_dict[study_id] = []

        metadata_dict[study_id].append(metadata)

    return metadata_dict


if __name__ == '__main__':
    
    dir_path = 'm:/magic/output/Pre_post_CT_XR_cohort/20220328072248/1003813_1_2_9c4f7c2e-cdfd-457e-9e24-7895bde42c9c/'
    metadata = read_metadata_single_dir(dir_path)

    root_dir = 'm:/magic/output/Pre_post_CT_XR_cohort/'
    metadata = read_metadata_root_dir(root_dir)

    metadata_file = 'm:/moshe/3d_prediction/results/3d_db/metadata.dat'
    utils.save_compressed_pickle(metadata, metadata_file)

    pass