from pathlib import Path

from mid.data import Annotation
from mid.data import MisRefreshDataset
from mid.tests import read_test_data, read_data


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def test_export_misrefresh_to_coco():

	# ann1, ann2, transform = read_data('MR04-001_M12_M6.dat')

	# ann_dict1, ann_dict2, img1, img2, pixel_spacing, units = read_test_data('MR04-019.dat')[0:6]
	# ann1 = Annotation(ann_dict1, pixel_spacing, units)
	# ann2 = Annotation(ann_dict2, pixel_spacing, units)

	test_data_path = Path(__file__).parents[1] / 'test_data'
	vert_dir = (test_data_path / 'mis_refresh_2d_features/vert_data').resolve().as_posix()
	rod_dir = (test_data_path / 'mis_refresh_2d_features/rod_data').resolve().as_posix()
	screw_dir = (test_data_path / 'mis_refresh_2d_features/screw_data').resolve().as_posix()
	dicom_dir = (test_data_path).resolve().as_posix()

	ds = MisRefreshDataset(vert_dir, rod_dir, screw_dir, dicom_dir)

	study_id_list = ds.get_study_id_list(key='dicom_df', col_name='StudyID')

	images_list = []
	annotations_list = []
	categories_list = []

	for study_id in study_id_list:

		# TODO: add filter by projection, and maybe other attributes

		# get all unique dicoms
		df = ds.filter_anns_df(ds.dataset['dicom_df'], study_id=study_id)
		dicom_path_list = sorted(df['dicom_path'].unique())

		for dicom_path in dicom_path_list:

			dicom_path = dicom_path_list[0]
			df_row = ds.filter_anns_df(ds.dataset['dicom_df'], dicom_path=dicom_path)

			projection = df_row['projection'].values[0]
			acquired = df_row['acquired'].values[0]
			bodyPos = df_row['bodyPos'].values[0]

			ann = ds.get_ann(study_id=study_id, projection=projection, body_pos=bodyPos, acquired=acquired, units='pixel', display=False)





		pass



	pass



if __name__ == '__main__':

	test_export_misrefresh_to_coco()

	pass