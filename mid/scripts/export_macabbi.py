from pathlib import Path
import numpy as np
import cv2
import shutil
import json
from tqdm import tqdm

from mid.data import MaccbiDataset, adjust_dynamic_range, simple_preprocssing
from mid.export import get_ann_categories, keypoints2bbox, keypoints2segmentation

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def export_maccabi_to_coco():

    anns_type = 'implant'  # one of {'implant', 'vert_implant'}
    vert_visibility_flag = 0 if (anns_type == 'implant') else 2
    n_max_study_id = 1  #-1
    img_processing_type = 'adjust_dynamic_range'
    # img_processing_type = 'clahe1'
    cfg_update = {'pixel_spacing_override': (1., 1.)}

    # load dataset
    data_path = Path('/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/')
    vert_file = (data_path / 'vert' / 'vert.parquet').resolve().as_posix()
    rod_file = (data_path / 'rod' / 'rod.parquet').resolve().as_posix()
    screw_file = (data_path / 'screw' / 'screw.parquet').resolve().as_posix()
    dicom_file = (data_path / 'dicom' / 'dicom.parquet').resolve().as_posix()
    ds = MaccbiDataset(vert_file=vert_file, rod_file=rod_file, screw_file=screw_file, dicom_file=dicom_file, cfg_update=cfg_update)

    output_dir = data_path.parent /  'output' / data_path.name / 'coco_dataset'
    if output_dir.is_dir():
        shutil.rmtree(output_dir)  # delete dir
    images_dir = output_dir / 'data'
    images_dir.mkdir(parents=True, exist_ok=True)

    # get study id list
    study_id_list = ds.get_study_id_list(key='dicom_df', col_name='StudyID')

    # convert to coco format
    images_list = []
    annotations_list = []
    categories_list, cat_id2name, cat_name2id = get_ann_categories()
    img_id = 0
    ann_id = 0
    for n, study_id in tqdm(enumerate(study_id_list), total=len(study_id_list)):

        if (n_max_study_id > 0) and (n >= n_max_study_id):
            break

        # get all unique dicoms
        df = ds.filter_anns_df(ds.dataset['dicom_df'], study_id=study_id)
        dicom_path_list = sorted(df['dicom_path'].unique())

        for dicom_path in dicom_path_list:

            try:
                img_id += 1

                df_row = ds.filter_anns_df(ds.dataset['dicom_df'], dicom_path=dicom_path)

                projection = df_row['projection'].values[0]
                acquired = df_row['acquired'].values[0]
                acquired_date = df_row['acquired_date'].values[0]
                bodyPos = df_row['bodyPos'].values[0]
                relative_file_path = df_row['relative_file_path'].values[0]

                ann = ds.get_ann(study_id=study_id, projection=projection, body_pos=bodyPos, acquired=acquired,
                                 acquired_date=acquired_date, relative_file_path=relative_file_path,
                                 units='mm', display=False)  # for maccabi data should use units 'mm' and pixel_space_override (1,1)

                # images
                img = ann.load_dicom()
                if img_processing_type == 'adjust_dynamic_range':
                    img = adjust_dynamic_range(img, vmax=255, dtype=np.uint8, min_max_type='img')  # TODO: maybe pass?
                elif img_processing_type == 'clahe1':
                    img = adjust_dynamic_range(img, vmax=1., dtype=np.float32, min_max_type='img')  # convert to float with range [0., 1.]
                    img = simple_preprocssing(img, process_type='clahe1', keep_input_dtype=True, display=False)
                    img = adjust_dynamic_range(img, vmax=255, dtype=np.uint8, min_max_type='img')  # convert to uint8 with range [0, 255]


                file_name = '{:09d}.jpg'.format(img_id)
                file_name_full = images_dir / file_name
                cv2.imwrite(file_name_full.resolve().as_posix(), img)
                img_dict = {'file_name': file_name,
                            'id': img_id,
                            'height': img.shape[0],
                            'width': img.shape[1],
                            'metadata': {'study_id': int(study_id),
                                         'projection': projection,
                                         'bodyPos': bodyPos,
                                         'acquired': acquired,
                                         # relative path ?
                                         }
                            }

                # annotations
                # keys = ann.get_keys(anns_type)
                keys = ann.get_keys(
                    'vert_implant')  # always get all vert and implant keys, but sometimes set vert visibility flag to 0
                for key in keys:

                    if key.startswith(ann.vert_prefix):
                        keypoints_row_index = 0
                        visibility_flag = vert_visibility_flag
                    elif key.startswith(ann.screw_prefix):
                        keypoints_row_index = 4
                        visibility_flag = 2  # always visible
                    elif key.startswith(ann.rod_prefix):
                        keypoints_row_index = 6
                        visibility_flag = 2  # always visible
                    else:
                        continue

                    ann_id += 1

                    category_id = 1

                    keypoints = ann[key].round(2)
                    segmentation = keypoints2segmentation(keypoints)
                    bbox = keypoints2bbox(keypoints)
                    num_keypoints = keypoints.shape[0]
                    area = bbox[2] * bbox[3]

                    # define keypoints in coco_spine_xr format:
                    #    [[x_vert_upper_start, y_vert_upper_start, v],
                    #     [x_vert_upper_end, y_vert_upper_end, v],
                    #     [x_vert_lower_end, y_vert_lower_end, v],
                    #     [x_vert_lower_start, y_vert_lower_start, v],
                    #     [x_screw_start, y_screw_start, v],
                    #     [x_screw_end, y_screw_end, v],
                    #     [x_rod_start, y_rod_start, v],
                    #     [x_rod_end, y_rod_end, v],
                    #    ]
                    #
                    # where v is visibility flag defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible
                    #
                    # vert anns: if start is on the left-hand side, then ann format is [left_top, right_top, right_bottom, left_bottom]

                    keypoints_coco = np.zeros((8, 3))  #
                    keypoints = np.concatenate((keypoints, visibility_flag * np.ones((keypoints.shape[0], 1))),
                                               axis=1)  # add visibility flag
                    keypoints_coco[keypoints_row_index:(keypoints_row_index + keypoints.shape[0]), :] = keypoints
                    keypoints_coco = keypoints_coco.flatten().tolist()

                    ann_dict = {'segmentation': segmentation,
                                'bbox': bbox,
                                'keypoints': keypoints_coco,
                                'num_keypoints': num_keypoints,
                                'area': area,
                                'iscrowd': 0,
                                'id': ann_id,
                                'image_id': img_id,
                                'category_id': category_id,
                                'name': key,
                                }

                    # update output lists
                    images_list.append(img_dict)
                    annotations_list.append(ann_dict)

                pass

            except:
                print('exception: skipping\n study_id: {}\n img_id: {}\n dicom_path: {}'.format(study_id, img_id, dicom_path))
                pass

    print('\n')
    print('----------------------------------')
    print('Total # of exported')
    print('study_ids: {}'.format(n))
    print('images: {}'.format(img_id))
    print('annotations: {}'.format(ann_id))
    print('----------------------------------')
    print('\n')

    # write ann file
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

    export_maccabi_to_coco()

    pass
