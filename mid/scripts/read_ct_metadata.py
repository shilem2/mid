from pathlib import Path
import pandas as pd
from mid.data import utils
from mid.data.read_3d_data import read_metadata_single_dir, read_metadata_root_dir, generate_metadata_df, filter_anns_df, generate_3d_meta_df
from mid.data import MaccbiDataset

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


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
    metadata = utils.load_compressed_pickle(metadata_file)

    pass


def generate_metadata_df_example():

    root_dir = 'm:/magic/output/Pre_post_CT_XR_cohort/'
    num_max = 30
    output_df_file = 'm:/moshe/3d_prediction/results/3d_db/metadata_df_{}.parquet'.format(num_max)

    generate_metadata_df(root_dir, pattern='**/Patient.json', num_max=num_max, output_df_file=output_df_file)

    pass

def read_xr_data():

    # data_path = Path('/mnt/magic_efs/moshe/implant_detection/data/2023-01-17_merged_data_v2/')
    data_path = Path('M:/moshe/data/2023-01-17_merged_data_v2/')
    vert_file = (data_path / 'vert' / 'vert.parquet').resolve().as_posix()
    rod_file = (data_path / 'rod' / 'rod.parquet').resolve().as_posix()
    screw_file = (data_path / 'screw' / 'screw.parquet').resolve().as_posix()
    dicom_file = (data_path / 'dicom' / 'dicom.parquet').resolve().as_posix()

    cfg_update_data = {'pixel_spacing_override': (1., 1.),  # None
                       'pairs_for_registration': {'acquired_date': 'different',
                                                  'skip_flipped_anns': False,
                                                  'latest_preop': True,
                                                  'latest_postop': False,
                                                  'projection': 'LT',
                                                  'body_pose': 'Neutral',
                                                  },
                       }

    dataset = MaccbiDataset(vert_file=vert_file, rod_file=rod_file, screw_file=screw_file, dicom_file=dicom_file, cfg_update=cfg_update_data)

    study_ids = dataset.get_study_id_list(key='vert_df')

    study_id = 1003813  # 1 ct
    study_id = 1023714  # 22 cts

    study_df = dataset.filter_study_id(study_id)
    df_xr = study_df.drop_duplicates('relative_file_path')

    metadata_df_file = 'm:/moshe/3d_prediction/results/3d_db/metadata_df_30.parquet'
    df_3d_full = pd.read_parquet(metadata_df_file)

    df_3d = filter_anns_df(df_3d_full, study_id=study_id)

    first_procedure_date_list = df_xr['first_procedure_date'].unique()
    if len(first_procedure_date_list):
        # FIXME: sort by date and take earliest
        pass
    else:
        first_procedure_date = first_procedure_date_list[0]

    first_procedure_date_str = first_procedure_date.strftime('%Y-%m-%d')

    df_3d['is_preop'] = df_3d.dcm_date.apply(lambda x: x.strftime('%Y-%m-%d') < first_procedure_date_str)
    df_3d['is_postop'] = df_3d.dcm_date.apply(lambda x: x.strftime('%Y-%m-%d') >= first_procedure_date_str)

    pass


def generate_3d_meta_df_example():

    meta_root_dir = 'm:/magic/output/Pre_post_CT_XR_cohort/'
    procedure_meta_file = 'm:/moshe/Maccabi_DB/postop_CT_WO_immediate.csv'
    num_max = 30
    output_dir_sfx = '_postop_ct_wo_immediate'
    output_df_file = 'm:/moshe/3d_prediction/results/3d_db/metadata_df_{}{}.parquet'.format(num_max if num_max > 0 else '', output_dir_sfx)

    df = generate_3d_meta_df(meta_root_dir, procedure_meta_file, output_df_file=output_df_file, pattern='**/Patient.json', num_max=num_max)

    pass


if __name__ == '__main__':

    # read_metadata_single_dir_example()
    # read_metadata_root_dir_example()
    # generate_metadata_df_example()
    # read_xr_data()
    generate_3d_meta_df_example()

    pass