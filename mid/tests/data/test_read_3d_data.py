from pathlib import Path
import pandas as pd

from mid.data.read_3d_data import read_metadata_single_dir, read_metadata_root_dir, generate_metadata_df, \
    filter_anns_df, read_procedures_file

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def test_read_metadata_single_dir():

    base_dir = Path(__file__).parents[2]
    dir_path = base_dir / 'tests' / 'test_data' / 'maccabi_ct_pipe' / '1003813_1_2_9c4f7c2e-cdfd-457e-9e24-7895bde42c9c'

    metadata, study_id = read_metadata_single_dir(dir_path, patient_file='Patient.json', study_file='Study.json', study_analysis_file='StudyAnalysis.json')

    assert isinstance(study_id, int)
    assert study_id == 1003813
    assert set(metadata.keys()) == {'dir_path', 'mongo_id', 'study_id', 'series_date'}
    assert metadata['study_id'] == 1003813
    assert metadata['mongo_id'] == 'caf295f3-23e7-4029-af15-f0033a55606b'
    assert metadata['series_date'].strftime('%Y-%m-%d') == '2006-08-10'

    pass


def test_read_metadata_root_dir():

    base_dir = Path(__file__).parents[2]
    root_dir = base_dir / 'tests' / 'test_data' / 'maccabi_ct_pipe'

    metadata_dict = read_metadata_root_dir(root_dir, pattern='**/Patient.json')

    assert set(metadata_dict.keys()) == {1003813, 1023714}
    assert len(metadata_dict[1003813]) == 1
    assert len(metadata_dict[1023714]) == 9
    assert metadata_dict[1023714][8]['series_date'].strftime('%Y-%m-%d') == '2008-01-12'

    pass


def test_generate_metadata_df():

    base_dir = Path(__file__).parents[2]
    root_dir = base_dir / 'tests' / 'test_data' / 'maccabi_ct_pipe'

    df = generate_metadata_df(root_dir, pattern='**/Patient.json', process_df_flag=True, num_max=-1, output_df_file=None)

    assert set(df.columns) == {'mongo_id', 'study_id', 'dcm_date', 'relative_dir_path', 'full_dir_path'}
    assert df.shape == (10, 5)

    df1 = filter_anns_df(df, study_id=1003813)
    assert df1.shape == (1, 5)

    df2 = filter_anns_df(df, study_id=1023714)
    assert df2.shape == (9, 5)

    pass


def test_calc_pre_post_op():

    base_dir = Path(__file__).parents[2]
    procedure_meta_file = base_dir / 'tests' / 'test_data' / 'all_postop_CT.csv'

    procedure_df = read_procedures_file(procedure_meta_file)

    root_dir = base_dir / 'tests' / 'test_data' / 'maccabi_ct_pipe_pre_post'
    # root_dir = 'm:/magic/output/Pre_post_CT_XR_cohort/20220328072248/'
    meta_df = generate_metadata_df(root_dir, pattern='**/Patient.json', process_df_flag=True, num_max=-1, output_df_file=None)

    df = meta_df.merge(procedure_df, on=['study_id'])

    df['is_preop'] = df['dcm_date'] < df['surgery_date']
    df['is_postop'] = df['dcm_date'] >= df['surgery_date']

    df[['study_id', 'surgery_date', 'dcm_date', 'is_preop', 'is_postop']]
    df[df['is_preop'] == True]

    assert df['is_preop'].sum() == 8
    assert df['is_postop'].sum() == 35



    df1 = filter_anns_df(df, study_id=1242288)


    pass

if __name__ == '__main__':

    # test_read_metadata_single_dir()
    # test_read_metadata_root_dir()
    # test_generate_metadata_df()
    test_calc_pre_post_op()

    pass