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
	# dataset_dir = '/mnt/magic_efs/moshe/implant_detection/data/output/2022-08-10_merged_data_v2/maccabi_6_study_ids_1_splits_skip_lr_flip/AP/0_6_study_ids/'
	# dataset_dir = '/mnt/magic_efs/moshe/implant_detection/data/output/2022-08-10_merged_data_v2/000_maccabi_all_study_ids_9_splits_skip_lr_flip/LT/1_101_study_ids/'
	dataset_dir = '/mnt/magic_efs/moshe/implant_detection/data/output/2022-08-10_merged_data_v2/001_maccabi_4_study_ids_2_splits_skip_lr_flip_add_verts_skip_lr_flip/LT/1_2_study_ids/'


	dataset_dir = Path(dataset_dir).resolve()
	labels_file = dataset_dir / 'coco_anns.json'

	dataset = fo.Dataset.from_dir(
		dataset_type=fo.types.COCODetectionDataset,
		label_types=["detections", "segmentations", "keypoints"],
		dataset_dir=dataset_dir,
		labels_path=labels_file)

	session = fo.launch_app(dataset)

	session.wait()

	pass


def mid_coco_tag_dataset():

	mid_dir = Path(__file__).parents[2].resolve()
	os.chdir(mid_dir)

	# dataset_dir = 'mid/tests/test_data/output/coco_dataset_12_images_implants_only/'
	# dataset_dir = 'mid/tests/test_data/output/coco_dataset_12_images_implants_only_verts_v_0/'
	# dataset_dir = '/home/shilem2/implant_detection/mid/mid/tests/test_data/output/coco_dataset_12_images_implants_only_verts_v_0'
	# dataset_dir = '/mnt/magic_efs/moshe/implant_detection/data/output/2022-08-10_merged_data_v2/coco_dataset_fixed_11_images/'
	# dataset_dir = '/mnt/magicdat_efs/moshe/implant_detection/data/output/2022-08-10_merged_data_v2/coco_dataset_5_study_ids_70_images/'
	dataset_dir = '/mnt/magic_efs/moshe/implant_detection/data/output/2022-08-10_merged_data_v2/maccabi_6_study_ids_1_splits_skip_lr_flip/LT/0_6_study_ids/'
	# dataset_dir = '/mnt/magic_efs/moshe/implant_detection/data/output/2022-08-10_merged_data_v2/maccabi_6_study_ids_1_splits_skip_lr_flip/AP/0_6_study_ids/'

	# dataset_name = Path(dataset_dir).name
	dataset_name = '.'.join(Path(dataset_dir).parts[len(Path(dataset_dir).parts)-4:])

	dataset_dir = Path(dataset_dir).resolve()
	labels_file = dataset_dir / 'coco_anns.json'

	# fo.delete_dataset(dataset_name)  # delete current dataset_name
	# [fo.delete_dataset(name) for name in fo.list_datasets()]  # delete all existing datasets

	if dataset_name in fo.list_datasets():
		dataset = fo.load_dataset(dataset_name)
	else:
		dataset = fo.Dataset.from_dir(
			dataset_type=fo.types.COCODetectionDataset,
			label_types=["detections", "segmentations", "keypoints"],
			dataset_dir=dataset_dir,
			labels_path=labels_file,
			name=dataset_name,
		)

		dataset.persistent = True

	session = fo.launch_app(dataset)

	# select images to filter on the app
	filter_ids = session.selected

	view_to_filter = dataset[filter_ids]

	for sample in view_to_filter:
		sample.tags.append('flip_lr')
		sample.save()

	export_dir = dataset_dir.as_posix() + '_tag_flip'
	dataset.export(
		export_dir=export_dir,
		dataset_type=fo.types.COCODetectionDataset,
	)

	session.wait()

	pass


if __name__ == '__main__':

	# fo_quickstart()
	# mmpose_coco_test()
	mid_coco_test()
	# mid_coco_tag_dataset()


	pass