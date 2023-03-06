import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import numpy as np

VERT_NAMES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
              'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
              'L1', 'L2', 'L3', 'L4', 'L5', 'L6',
              'S1', 'S2', 'S3', 'S4', 'S5',
              ]

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
        'series_date': parse_date_string(study['seriesDate']),
        'dir_path': dir_path.absolute().as_posix(),
    }

    study_analysis_data = extract_study_analysis_data(study_analysis)
    metadata.update(study_analysis_data)

    return metadata, study_id

def parse_date_string(date_str):
    """
    Convert string to datetime object.
    """

    try:
        dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%z')
    except:
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%fz')  # with micro-seconds %f
        except:
            dt = date_str

    return dt



def extract_study_analysis_data(study_analysis, merge_type='union'):

    # currently labels are found in 2 places inside study_analysis
    vert_labels_metadata = [v['label'] for v in study_analysis['metadata']['vertInfo']]  # RTC's calculations
    vert_labels_vert_list = [v['label'] for v in study_analysis['labels']['vertList']]  # Kfir's calculations

    assert merge_type in ['union', 'intersection']
    if merge_type == 'union':
        vert_labels = list(set(vert_labels_metadata) | set(vert_labels_vert_list))
    elif merge_type == 'intersection':
        vert_labels = list(set(vert_labels_metadata) & set(vert_labels_vert_list))

    vert_labels = sort_keys_by_names(vert_labels)

    study_analysis_data = {
        'vert_labels': vert_labels,
    }

    return study_analysis_data


def sort_keys_by_names(keys, wanted_order=VERT_NAMES):
    """Sort keys by wanted order.
    """
    indices_ordered = list(range(len(wanted_order)))
    zipped_sorted_ind_vert = list(zip(indices_ordered, wanted_order))
    indices = sorted([ind for (ind, key) in zipped_sorted_ind_vert if key in keys])  # indices of input keys
    keys_ordered = [wanted_order[ind] for ind in indices]

    return keys_ordered


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


def generate_metadata_df(root_dir_list, pattern='**/Patient.json', process_df_flag=True, num_max=-1, output_df_file=None, subdirs_list=[]):
    """
    Read metadata of CT pipe output - all directories under root_dir
    returns DataFrame.
    """

    if not isinstance(root_dir_list, list):
        root_dir_list = [root_dir_list]


    srs = []
    n = 0
    for root_dir in tqdm(root_dir_list, desc='root dir'):

        paths = Path(root_dir).rglob(pattern)
        # path_list = list(paths)

        for path in tqdm(paths, desc='path'):

            metadata, study_id = read_metadata_single_dir(path.parent)
            sr = pd.Series(metadata)
            srs.append(sr)

            if (num_max > 0) and (n >= (num_max - 1)):
                break

            n += 1

    df = pd.concat(srs, axis=1).transpose()

    if process_df_flag:
        df = process_df(df)

    if output_df_file is not None:
        df.to_parquet(output_df_file)

    return df


def process_df(df, relative_path_start=4, out_cols=['study_id', 'mongo_id', 'full_dir_path', 'relative_dir_path', 'dcm_date', 'vert_labels']):
    """
    Process df.
    """

    df['relative_dir_path'] = df['dir_path'].apply(lambda x: '/'.join(x.split('/')[relative_path_start:]))
    df.rename(columns={'dir_path': 'full_dir_path', 'series_date': 'dcm_date'}, inplace=True)
    df['dcm_date'] = pd.to_datetime(df['dcm_date'], errors='coerce').dt.date

    df = df[out_cols]

    return df



def read_procedures_file(file_path, out_cols=['study_id', 'surgery_date']):

    df = pd.read_csv(file_path)

    df['study_id'] = df['Patient'].astype(int)
    df['surgery_date'] = pd.to_datetime(df['Surgery Date'], format='%m/%d/%Y', errors='coerce').dt.date

    df = df[out_cols]

    return df


def generate_3d_meta_df(meta_root_dir, procedure_meta_file, output_df_file=None, pattern='**/Patient.json', num_max=-1):
    """
    Generate unified metadata df, contains both metadata of CT pipe output, procedure date and pre/post-op flags.
    """

    procedure_df = read_procedures_file(procedure_meta_file)

    meta_df = generate_metadata_df(meta_root_dir, pattern=pattern, process_df_flag=True, num_max=num_max, output_df_file=output_df_file)

    df = meta_df.merge(procedure_df, on=['study_id'])

    # calculate interesting values
    df['is_preop'] = df['dcm_date'] < df['surgery_date']
    df['is_postop'] = df['dcm_date'] >= df['surgery_date']
    df['days_after_surgery'] = df['dcm_date'] - df['surgery_date']
    df['days_after_surgery'] = df['days_after_surgery'].apply(lambda x: x.days)

    if output_df_file is not None:
        df.to_parquet(Path(output_df_file).with_suffix('.parquet'))
        df.to_csv(Path(output_df_file).with_suffix('.csv'))

    # debug - save unique study_ids
    np.savetxt(Path(output_df_file).parent / 'study_ids_procedure.csv', procedure_df.study_id.unique(), delimiter='\n'); print(len(procedure_df.study_id.unique()))
    np.savetxt(Path(output_df_file).parent / 'study_ids_ct_pipe.csv', meta_df.study_id.unique(), delimiter='\n'); print(len(meta_df.study_id.unique()))
    np.savetxt(Path(output_df_file).parent / 'study_ids_intersection.csv', df.study_id.unique(), delimiter='\n'); print(len(df.study_id.unique()))
    np.savetxt(Path(output_df_file).parent / 'study_ids_diff.csv', np.array(list(set(procedure_df.study_id.unique()) - set(df.study_id.unique()))), delimiter='\n'); print(len(np.array(list(set(procedure_df.study_id.unique()) - set(df.study_id.unique())))))

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

