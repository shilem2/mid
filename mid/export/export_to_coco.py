import numpy as np


def get_ann_categories():

	categories = [
		{'id': 1,
		 'name': 'vert',
		 },
		{'id': 2,
		 'name': 'screw',
		 },
		{'id': 3,
		 'name': 'rod',
		 },
	]

	return categories


def keypoints2bbox(keypoints):

	left = keypoints[:, 0].min()
	right = keypoints[:, 0].max()
	top = keypoints[:, 1].min()
	bottom = keypoints[:, 1].max()
	width = right - left
	height = bottom - top

	bbox = np.array([left, top, width, height])

	return bbox

def keypoints2segmentation(keypoints):
	"""
	keypoints: ndarray of shape (N, 2) where each row is (x, y) pair.
	"""

	segmentation = np.concatenate((keypoints.flatten(), keypoints[0, :]))  # close polygon by first pair of coordinates
	return segmentation