from pathlib import Path
import pandas as pd

from mid.data.read_3d_data import read_metadata_single_dir, read_metadata_root_dir, generate_metadata_df, \
    filter_anns_df, generate_3d_meta_df

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def test_read_metadata_single_dir():

    base_dir = Path(__file__).parents[2]
    dir_path = base_dir / 'tests' / 'test_data' / 'maccabi_ct_pipe' / '1003813_1_2_9c4f7c2e-cdfd-457e-9e24-7895bde42c9c'

    metadata, study_id = read_metadata_single_dir(dir_path, patient_file='Patient.json', study_file='Study.json', study_analysis_file='StudyAnalysis.json')

    assert isinstance(study_id, int)
    assert study_id == 1003813
    assert set(metadata.keys()) == {'dir_path', 'mongo_id', 'study_id', 'series_date', 'vert_labels'}
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

    assert set(df.columns) == {'mongo_id', 'study_id', 'dcm_date', 'relative_dir_path', 'full_dir_path', 'vert_labels'}
    assert df.shape == (10, 6)

    df1 = filter_anns_df(df, study_id=1003813)
    assert df1.shape == (1, 6)

    df2 = filter_anns_df(df, study_id=1023714)
    assert df2.shape == (9, 6)

    pass


def test_generate_3d_meta_df():

    base_dir = Path(__file__).parents[2]
    procedure_meta_file = base_dir / 'tests' / 'test_data' / 'postop_CT_all.csv'
    meta_root_dir = base_dir / 'tests' / 'test_data' / 'maccabi_ct_pipe_pre_post'
    # meta_root_dir = 'm:/magic/output/Pre_post_CT_XR_cohort/20220328072248/'

    df = generate_3d_meta_df(meta_root_dir, procedure_meta_file, output_df_file=None)

    assert df.shape == (43, 10)
    assert df['is_preop'].sum() == 8
    assert df['is_postop'].sum() == 35

    df1 = filter_anns_df(df, study_id=1242288)
    assert df1.shape == (27, 10)
    assert df1.loc[16, 'days_after_surgery'] == -60
    assert df1.loc[42, 'days_after_surgery'] == 618

    pass


if __name__ == '__main__':

    # test_read_metadata_single_dir()
    # test_read_metadata_root_dir()
    # test_generate_metadata_df()
    test_generate_3d_meta_df()

    pass