import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from mid.data import utils


def read_metadata_single_dir(dir_path, patient_file='Patient.json', study_file='Study.json',
                             study_analysis_file='StudyAnalysis.json'):
    dir_path = Path(dir_path)

    patient = json.load(open(dir_path / patient_file))
    study = json.load(open(dir_path / study_file))
    study_analysis = json.load(open(dir_path / study_analysis_file))

    study_id = patient['patientId']  # Maccabi study id

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


def generate_metadata_df(root_dir, pattern='**/Patient.json', process_df_flag=True, num_max=-1, output_df_file=None):

    paths = Path(root_dir).rglob(pattern)
    # path_list = list(paths)

    srs = []
    for n, path in tqdm(enumerate(paths)):

        metadata, study_id = read_metadata_single_dir(path.parent)
        sr = pd.Series(metadata)
        srs.append(sr)

        if (num_max > 0) and (n >= (num_max - 1)):
            break

    df = pd.concat(srs, axis=1).transpose()

    if process_df_flag:
        df = process_df(df)

    if output_df_file is not None:
        df.to_parquet(output_df_file)

    return df


def process_df(df, relative_path_start=4, out_cols=['study_id', 'mongo_id', 'full_dir_path', 'relative_dir_path', 'dcm_date']):

    df['study_id'] = df.study_id.astype(int)
    df['relative_dir_path'] = df['dir_path'].apply(lambda x: '/'.join(x.split('/')[relative_path_start:]))
    df['dcm_date'] = pd.to_datetime(df['series_date'], format='%Y-%m-%dT%H:%M:%S')
    df['dcm_date'] = df['dcm_date'].apply(lambda x: x.date())
    df.rename(columns={'dir_path': 'full_dir_path'}, inplace=True)

    df = df[out_cols]

    return df


def filter_anns_df(df, study_id=None, mongo_id=None, relative_file_path=None, dcm_date=None):
    """
    Filter annotations data frame.
    At least on of the arguments other than vert_df should be given.
    None value means that filtering does not consider that argument.
    """

    inds = ((study_id is None) or (df.study_id == study_id)) & \
           ((mongo_id is None) or (df.mongo_id == mongo_id)) & \
           ((relative_file_path is None) or (df.relative_file_path == relative_file_path)) & \
           ((dcm_date is None) or (df.dcm_date == dcm_date))

    df_out = df[inds]

    return df_out

