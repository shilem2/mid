import numpy as np


def get_ann_categories():

	categories = [
		{'id': 1,
		 'name': 'spine_xr',
		 },
		# {'id': 1,
		#  'name': 'vert',
		#  },
		# {'id': 2,
		#  'name': 'screw',
		#  },
		# {'id': 3,
		#  'name': 'rod',
		#  },
	]

	id2name = {}
	name2id = {}
	for cat in categories:
		id2name[cat['id']] = cat['name']
		name2id[cat['name']] = cat['id']

	return categories, id2name, name2id


def keypoints2bbox(keypoints):

	left = keypoints[:, 0].min()
	right = keypoints[:, 0].max()
	top = keypoints[:, 1].min()
	bottom = keypoints[:, 1].max()
	width = right - left
	height = bottom - top

	# enforce minimal values
	left = np.maximum(left, 0.)
	top = np.maximum(top, 0.)
	width = np.maximum(width, 1.)
	height = np.maximum(height, 1.)

	bbox = np.array([left, top, width, height]).flatten().round(2).tolist()

	return bbox

def keypoints2segmentation(keypoints):
	"""
	Generate COCO format segmentation from a set of keypoints.

	Parameters
	----------
    keypoints : ndarray
        array of shape (N, 2) where each row is (x, y) pair.

    Returns
    -------
    segmentation : list
        Flattened list of coordinates given as [x0, y0, x1, y1, ...].
        Coco segmentation format is list of lists, since segmentation of single object can be divided to several
        polygons (e.g. due to occlusions).
	"""
	segmentation = [np.concatenate((keypoints.flatten(), keypoints[0, :])).tolist()]  # close polygon by first pair of coordinates
	return segmentation