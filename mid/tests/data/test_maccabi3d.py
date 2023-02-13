from pathlib import Path

from mid.data import Maccbi3dDataset


def test_init_Maccabi3d():


    base_dir = Path(__file__).parents[2]
    meta_file = base_dir / 'tests' / 'test_data' / 'metadata_df_postop_ct_wo_immediate.parquet'

    dataset = Maccbi3dDataset(meta_file=meta_file)

    study_id_list = dataset.get_study_id_list()

    assert len(study_id_list) == 41

    dataset.dataset['meta_df'].study_id.unique()

    pass

if __name__ == '__main__':

    test_init_Maccabi3d()

    pass