from pathlib import Path
import numpy as np
import cv2
import shutil
import json
from tqdm import tqdm

from mid.data import MisRefreshDataset, adjust_dynamic_range, utils
from mid.export import get_ann_categories, keypoints2bbox, keypoints2segmentation

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def test_export_misrefresh_to_coco():


	# load dataset
	test_data_path = Path(__file__).parents[1] / 'test_data'
	vert_dir = (test_data_path / 'mis_refresh_2d_features/vert_data').resolve().as_posix()
	rod_dir = (test_data_path / 'mis_refresh_2d_features/rod_data').resolve().as_posix()
	screw_dir = (test_data_path / 'mis_refresh_2d_features/screw_data').resolve().as_posix()
	dicom_dir = (test_data_path).resolve().as_posix()
	ds = MisRefreshDataset(vert_dir, rod_dir, screw_dir, dicom_dir)

	output_dir = test_data_path / 'output' / 'coco_dataset'
	shutil.rmtree(output_dir)
	images_dir = output_dir / 'images'
	images_dir.mkdir(parents=True, exist_ok=True)

	# get study id list
	study_id_list = ds.get_study_id_list(key='dicom_df', col_name='StudyID')

	# convert to coco format
	images_list = []
	annotations_list = []
	categories_list = get_ann_categories()
	img_id = 0
	ann_id = 0
	for study_id in study_id_list:

		# get all unique dicoms
		df = ds.filter_anns_df(ds.dataset['dicom_df'], study_id=study_id)
		dicom_path_list = sorted(df['dicom_path'].unique())

		for dicom_path in dicom_path_list:

			img_id += 1

			df_row = ds.filter_anns_df(ds.dataset['dicom_df'], dicom_path=dicom_path)

			projection = df_row['projection'].values[0]
			acquired = df_row['acquired'].values[0]
			bodyPos = df_row['bodyPos'].values[0]

			ann = ds.get_ann(study_id=study_id, projection=projection, body_pos=bodyPos, acquired=acquired, units='pixel', display=False)

			# images
			img = ann.load_dicom()
			img = adjust_dynamic_range(img, vmax=255, dtype=np.uint8, min_max_type='img')  # TODO: maybe pass?
			file_name = '{:09d}.jpg'.format(img_id)
			file_name_full = images_dir / file_name
			cv2.imwrite(file_name_full.resolve().as_posix(), img)
			img_dict = {'file_name': file_name,
						'id': img_id,
						'height': img.shape[0],
						'width': img.shape[1],
						}

			images_list.append(img_dict)

			# annotations
			# ktype in ['vert', 'rod', 'screw', 'icl', 'femur', 'all']
			# verts
			keys= ann.get_keys('vert')
			for key in keys:

				ann_id += 1

				keypoints = ann[key].round(2)
				segmentation = keypoints2segmentation(keypoints).flatten().tolist()
				bbox = keypoints2bbox(keypoints).flatten().round(2).tolist()
				num_keypoints = keypoints.shape[0]
				area = bbox[2] * bbox[3]
				category_id = 1  # 1:vert, 2:screw, 3:rod
				keypoints = np.concatenate((keypoints, 2 * np.ones((keypoints.shape[0], 1))), axis=1).flatten().tolist()  # coco keypoints format is (x,y,v), where v is visibility flag defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible

				ann_dict = {'segmentation': segmentation,
							'bbox': bbox,
							'keypoints': keypoints,
							'num_keypoints': num_keypoints,
							'area': area,
							'iscrowd': 0,
							'id': ann_id,
							'image_id': img_id,
							'category_id': category_id,
							'category_name': 'vert',
							'name': key,
							}

				annotations_list.append(ann_dict)

			# screws
			keys = ann.get_keys('screw')
			for key in keys:

				ann_id += 1

				keypoints = ann[key].round(2)
				segmentation = keypoints2segmentation(keypoints).flatten().tolist()
				bbox = keypoints2bbox(keypoints).flatten().round(2).tolist()
				num_keypoints = keypoints.shape[0]
				area = bbox[2] * bbox[3]
				category_id = 2  # 1:vert, 2:screw, 3:rod
				keypoints = np.concatenate((keypoints, 2 * np.ones((keypoints.shape[0], 1))), axis=1).flatten().tolist()  # coco keypoints format is (x,y,v), where v is visibility flag defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible

				ann_dict = {'segmentation': segmentation,
							'bbox': bbox,
							'keypoints': keypoints,
							'num_keypoints': num_keypoints,
							'area': area,
							'iscrowd': 0,
							'id': ann_id,
							'image_id': img_id,
							'category_id': category_id,
							'category_name': 'screw',
							'name': key,
							}

				annotations_list.append(ann_dict)


		pass

	data = {'images': images_list,
			'annotations': annotations_list,
			'categories': categories_list
			}
	coco_file_name = 'coco_anns.json'
	coco_file_name_full = output_dir / coco_file_name

	with open(coco_file_name_full, 'w', encoding='utf-8') as f:
		# json.dump(data, f, default=utils.convert_array_to_json, ensure_ascii=False, indent=4)
		json.dump(data, f, ensure_ascii=False, indent=4)



	pass



if __name__ == '__main__':

	test_export_misrefresh_to_coco()

	pass