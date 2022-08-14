import os
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


def fo_quickstart():

	dataset = foz.load_zoo_dataset("quickstart")
	session = fo.launch_app(dataset)

	pass

def mmpose_coco_test():

	labels_file = r"/Users/shilem2/data/human_pose_estimation/coco_overfit/all/test_coco.json"
	dataset_file = r"/Users/shilem2/data/human_pose_estimation/coco_overfit/all_fiftyone"  # folder

	dataset = fo.Dataset.from_dir(
		dataset_type=fo.types.COCODetectionDataset,
		label_types=["detections", "segmentations", "keypoints"],
		dataset_dir=dataset_file,
		labels_path=labels_file)

	session = fo.launch_app(dataset)

	pass

def mid_coco_test():

	mid_dir = Path(__file__).parents[2].resolve()
	os.chdir(mid_dir)

	# dataset_dir = 'mid/tests/test_data/output/coco_dataset_12_images_implants_only/'
	# dataset_dir = 'mid/tests/test_data/output/coco_dataset_12_images_implants_only_verts_v_0/'
	dataset_dir = '/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/output/coco_dataset_2_study_ids/'

	dataset_dir = Path(dataset_dir).resolve()
	labels_file = dataset_dir / 'coco_anns.json'

	dataset = fo.Dataset.from_dir(
		dataset_type=fo.types.COCODetectionDataset,
		label_types=["detections", "segmentations", "keypoints"],
		dataset_dir=dataset_dir,
		labels_path=labels_file)

	session = fo.launch_app(dataset)

	pass


if __name__ == '__main__':

	# fo_quickstart()
	# mmpose_coco_test()
	mid_coco_test()

	pass