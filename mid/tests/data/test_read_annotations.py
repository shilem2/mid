import os.path
from pathlib import Path
from pytest import approx
import numpy as np

from mid.data.read_annotations import filter_anns_df, get_dicom_paths, get_scan_anns, load_anns_df


def test_get_dicom_paths_root_dir():

    root_dir = (Path(__file__).parent.parent / 'test_data/MR04-001').resolve()
    json_pattern = '**/XR/**/*det.json'
    dicom_pattern = '*.dcm'

    df = get_dicom_paths(root_dir, json_pattern, dicom_pattern)

    assert df.shape == (12, 5)

    data_row = filter_anns_df(df, study_id='MR04-001', projection='LT', body_pos='Flexion', acquired='Month 12')
    dicom_path = Path(data_row['dicom_path'].values[0])

    assert dicom_path.name == 'Flexion.dcm'
    assert dicom_path.parent.as_posix().split(os.sep)[-1] == 'NA - 1'


def test_load_all_anns():

    # parameters
    root_dir = (Path(__file__).parent.parent / 'test_data/mis_refresh_2d_features').resolve()
    vert_dir = root_dir / 'vert_data'
    rod_dir = root_dir / 'rod_data'
    screw_dir = root_dir / 'screw_data'
    dicom_dir = root_dir.parent / 'MR04-001'
    units = 'pixel'

    # read annotations
    vert_df = load_anns_df(vert_dir)
    rod_df = None  #load_anns_df(rod_dir)  # commented out since get_scan_anns() code has been updated for Maccabi dataset, but test data is MisRefresh - and there are some minor changes
    screw_df = None  #load_anns_df(screw_dir)
    dicom_df = get_dicom_paths(dicom_dir)

    ann = get_scan_anns(vert_df=vert_df, rod_df=rod_df, screw_df=screw_df, dicom_df=dicom_df,
                        study_id='MR04-001', projection='LT', body_pos='Neutral', acquired='Week 3-6',
                        units=units,
                        display=False)

    assert len(ann) == 7  #13
    assert ann.pixel_spacing == approx(np.array([1., 1.]))
    ann.change_units('mm')
    assert ann['L1'] == approx(np.array([[662.5, 310. ],
                                         [821.5, 237.5],
                                         [886.5, 375.5],
                                         [722.5, 438. ]]))

    pass


if __name__ == '__main__':

    # test_get_dicom_paths_root_dir()
    test_load_all_anns()

    print('Done!')