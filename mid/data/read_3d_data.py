import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime


def read_metadata_single_dir(dir_path, patient_file='Patient.json', study_file='Study.json', study_analysis_file='StudyAnalysis.json'):
    """
    Read metadata of CT pipe output - single directory
    """

    dir_path = Path(dir_path)

    patient = json.load(open(dir_path / patient_file))
    study = json.load(open(dir_path / study_file))
    study_analysis = json.load(open(dir_path / study_analysis_file))

    study_id = int(patient['patientId'])  # Maccabi study id

    metadata = {
        'study_id': study_id,
        'mongo_id': study['patientId'],  # Mongo DB id
        'series_date': datetime.strptime(study['seriesDate'], '%Y-%m-%dT%H:%M:%Sz').date(),  # convert string to datetime object, take only date
        'dir_path': dir_path.absolute().as_posix(),
        # 'study_analysis': study_analysis,
    }

    return metadata, study_id


def read_metadata_root_dir(root_dir, pattern='**/Patient.json'):
    """
    Read metadata of CT pipe output - all directories under root_dir
    this function returns a dict, it is recommended to use generate_metadata_df() which returns DataFrame.
    """


    paths = Path(root_dir).rglob(pattern)
    # path_list = list(paths)  # takes long time

    metadata_dict = {}
    for path in tqdm(paths):

        metadata, study_id = read_metadata_single_dir(path.parent)

        if study_id not in metadata_dict:
            metadata_dict[study_id] = []

        metadata_dict[study_id].append(metadata)

    return metadata_dict


def generate_metadata_df(root_dir, pattern='**/Patient.json', process_df_flag=True, num_max=-1, output_df_file=None):
    """
    Read metadata of CT pipe output - all directories under root_dir
    returns DataFrame.
    """

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
    """
    Process df.
    """

    df['relative_dir_path'] = df['dir_path'].apply(lambda x: '/'.join(x.split('/')[relative_path_start:]))
    df.rename(columns={'dir_path': 'full_dir_path', 'series_date': 'dcm_date'}, inplace=True)

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

def read_procedures_file(file_path, out_cols=['study_id', 'surgery_date']):

    df = pd.read_csv(file_path)

    df['study_id'] = df['Patient'].astype(int)
    df['surgery_date'] = pd.to_datetime(df['Surgery Date'], format='%m/%d/%Y')

    df = df[out_cols]

    return df

