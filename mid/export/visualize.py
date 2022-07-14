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

	# dataset_dir = '/Users/shilem2/OneDrive - Medtronic PLC/projects/mid/mid/tests/test_data/output/coco_dataset_12_images/'
	dataset_dir = '/Users/shilem2/OneDrive - Medtronic PLC/projects/mid/mid/tests/test_data/output/coco_dataset_1_image/'

	dataset_dir = Path(dataset_dir)
	labels_file = dataset_dir / 'coco_anns.json'
	dataset_file = dataset_dir

	dataset = fo.Dataset.from_dir(
		dataset_type=fo.types.COCODetectionDataset,
		label_types=["detections", "segmentations", "keypoints"],
		dataset_dir=dataset_file,
		labels_path=labels_file)

	session = fo.launch_app(dataset)

	pass


if __name__ == '__main__':

	# fo_quickstart()
	# mmpose_coco_test()
	mid_coco_test()

	pass