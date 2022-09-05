from pathlib import Path
import numpy as np
import cv2
import shutil
import json
from tqdm import tqdm
from pprint import pprint

from mid.data import MaccbiDataset, adjust_dynamic_range, simple_preprocssing
from mid.export import get_ann_categories, keypoints2bbox, keypoints2segmentation

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)


def export_maccabi_to_coco():

    anns_type = 'vert_implant'  #'implant'  # one of {'implant', 'vert_implant'}
    vert_visibility_flag = 0 if (anns_type == 'implant') else 2
    n_max_study_id = -1
    img_processing_type = 'adjust_dynamic_range'
    # img_processing_type = 'clahe1'
    cfg_update = {'pixel_spacing_override': (1., 1.)}
    skip_flipped_anns = True  # some of the annotations are horizontally flipped
    projection_list = ['LT', 'AP']
    n_split_list = 9

    # 001
    # output_dir_prefix = '001_'  # ''
    # output_dir_suffix = '_with_verts'  # ''
    # 002
    # output_dir_prefix = '002_'  # ''
    # output_dir_suffix = '_clahe1'  # ''
    # 003
    # output_dir_prefix = '003_'  # ''
    # output_dir_suffix = '_clahe1_with_verts'  # ''
    # 004
    output_dir_prefix = '004_'  # ''
    output_dir_suffix = '_with_verts_and_pixel_spacing'  # ''

    # load dataset
    data_path = Path('/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/')
    vert_file = (data_path / 'vert' / 'vert.parquet').resolve().as_posix()
    rod_file = (data_path / 'rod' / 'rod.parquet').resolve().as_posix()
    screw_file = (data_path / 'screw' / 'screw.parquet').resolve().as_posix()
    dicom_file = (data_path / 'dicom' / 'dicom.parquet').resolve().as_posix()
    ds = MaccbiDataset(vert_file=vert_file, rod_file=rod_file, screw_file=screw_file, dicom_file=dicom_file, cfg_update=cfg_update)

    skip_flipped_anns = skip_flipped_anns and ('x_sign' in ds.dataset['dicom_df'].columns)  # use skip flag only if dicom_df has 'x_sign' columns

    n_max_str = 'all' if n_max_study_id == -1 else n_max_study_id
    output_dir_base_name = '{}maccabi_{}_study_ids_{}_splits{}'.format(output_dir_prefix, n_max_str, n_split_list, output_dir_suffix)
    if skip_flipped_anns:
        output_dir_base_name += '_skip_lr_flip'
    output_dir_base = data_path.parent / 'output' / data_path.name / output_dir_base_name
    if output_dir_base.is_dir():
        shutil.rmtree(output_dir_base)  # delete dir

    # get study id list
    study_id_list_all = ds.get_study_id_list(key='dicom_df', col_name='StudyID')

    if (n_max_study_id > 0) and (n_max_study_id < len(study_id_list_all)):
        study_id_list_all = study_id_list_all[:n_max_study_id]

    if n_split_list > 1:
        study_id_list_of_lists = split_list(study_id_list_all, n_split_list, shuffle=False)
    else:
        study_id_list_of_lists = [study_id_list_all]

    for projection in projection_list:
        summary_list = []
        for n, study_id_list in enumerate(study_id_list_of_lists):
            study_id_list_str = '{}_{}_study_ids'.format(n, len(study_id_list))
            output_dir = output_dir_base / projection / study_id_list_str
            summary = export_study_id_list(ds, study_id_list, output_dir, projection, skip_flipped_anns, img_processing_type, vert_visibility_flag)
            summary_list.append(summary)

        summary_total = process_summaries(summary_list)

        print('-------------------------------')
        print('{} Summary:'.format(projection))
        pprint(summary_total)
        print('-------------------------------')

        # summary total
        summary_file_name = 'summary_total.json'
        summary_file_name_full = output_dir_base / projection / summary_file_name
        with open(summary_file_name_full, 'w', encoding='utf-8') as f:
            json.dump(summary_total, f, ensure_ascii=False, indent=4)

        # summary list
        summary_file_name = 'summaries.json'
        summary_file_name_full = output_dir_base / projection / summary_file_name
        with open(summary_file_name_full, 'w', encoding='utf-8') as f:
            json.dump(summary_list, f, ensure_ascii=False, indent=4)

    pass

def process_summaries(summary_list):

    summary_total = {}
    for summary in summary_list:
        for key, val in summary.items():
            if isinstance(val, int):
                if key not in summary_total.keys():
                    summary_total[key] = val
                else:
                    summary_total[key] += val

    return summary_total


def split_list(list_in, n, shuffle=False):

    if shuffle:
        np.random.shuffle(list_in)  # shuffle inplace

    list_out = np.array_split(list_in, n)
    list_out = [l.tolist() for l in list_out if len(l) > 0]

    return list_out

def export_study_id_list(ds, study_id_list, output_dir, projection, skip_flipped_anns=True, img_processing_type='adjust_dynamic_range', vert_visibility_flag=0):

    if output_dir.is_dir():
        shutil.rmtree(output_dir)  # delete dir
    images_dir = output_dir / 'data'
    images_dir.mkdir(parents=True, exist_ok=True)

    # convert to coco format
    images_list = []
    annotations_list = []
    categories_list, cat_id2name, cat_name2id = get_ann_categories()
    img_id = 0
    ann_id = 0
    ann_vert_counter = 0
    ann_screw_counter = 0
    ann_rod_counter = 0
    for n, study_id in tqdm(enumerate(study_id_list), total=len(study_id_list)):

        # get all unique dicoms
        df = ds.filter_anns_df(ds.dataset['dicom_df'], study_id=study_id, projection=projection)
        dicom_path_list = sorted(df['dicom_path'].unique())

        for dicom_path in dicom_path_list:

            try:

                df_row = ds.filter_anns_df(ds.dataset['dicom_df'], dicom_path=dicom_path)

                if skip_flipped_anns and (df_row['x_sign'].values[0] == -1):  # skip flipped anns
                    continue

                img_id += 1

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
                pixel_spacing = ann.metadata['pixel_spacing_orig'] if 'pixel_spacing_orig' in ann.metadata else ann.pixel_spacing
                img_dict = {'file_name': file_name,
                            'id': img_id,
                            'height': img.shape[0],
                            'width': img.shape[1],
                            'metadata': {'study_id': int(study_id),
                                         'projection': projection,
                                         'bodyPos': bodyPos,
                                         'acquired': acquired,
                                         'acquired_date': acquired_date,
                                         'relative path': relative_file_path,
                                         'full_path': dicom_path,
                                         'pixel_spacing': pixel_spacing.tolist(),
                                         }
                            }

                images_list.append(img_dict)

                # annotations
                # keys = ann.get_keys(anns_type)
                keys = ann.get_keys('vert_implant')  # always get all vert and implant keys, but sometimes set vert visibility flag to 0
                for key in keys:

                    if key.startswith(ann.vert_prefix):
                        keypoints_row_index = 0
                        visibility_flag = vert_visibility_flag
                        ann_vert_counter += 1
                    elif key.startswith(ann.screw_prefix):
                        keypoints_row_index = 4
                        visibility_flag = 2  # always visible
                        ann_screw_counter += 1
                    elif key.startswith(ann.rod_prefix):
                        keypoints_row_index = 6
                        visibility_flag = 2  # always visible
                        ann_rod_counter += 1
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
                    keypoints = np.concatenate((keypoints, visibility_flag * np.ones((keypoints.shape[0], 1))), axis=1)  # add visibility flag
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
                    annotations_list.append(ann_dict)

                pass

            except:
                print('exception: skipping\n study_id: {}\n img_id: {}\n dicom_path: {}'.format(study_id, img_id, dicom_path))
                pass

    summary_dict = {'output_dir': output_dir.as_posix(),
                    'projection': projection,
                    'study_ids': n + 1,  # +1 since enumerate starts at 0
                    'images': img_id,
                    'annotations': ann_id,
                    'anns_vert': ann_vert_counter,
                    'anns_screw': ann_screw_counter,
                    'anns_rod': ann_rod_counter,
                    }

    print('\n')
    print('----------------------------------')
    print('Export Summary:')
    pprint(summary_dict)
    print('----------------------------------')
    print('\n')

    summary_file_name = 'summary.json'
    summary_file_name_full = output_dir / summary_file_name
    with open(summary_file_name_full, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=4)

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

    return summary_dict


def visualize_single_example():

    study_id = 21455
    projection = 'AP'
    bodyPos = 'Neutral'
    acquired = '2009_10'
    acquired_date = 'Month 12'
    relative_file_path = '21455/XR/2009_10/AP/Neutral'

    display = True

    cfg_update = {'pixel_spacing_override': (1., 1.)}


    data_path = Path('/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/')
    vert_file = (data_path / 'vert' / 'vert.parquet').resolve().as_posix()
    rod_file = (data_path / 'rod' / 'rod.parquet').resolve().as_posix()
    screw_file = (data_path / 'screw' / 'screw.parquet').resolve().as_posix()
    dicom_file = (data_path / 'dicom' / 'dicom.parquet').resolve().as_posix()
    ds = MaccbiDataset(vert_file=vert_file, rod_file=rod_file, screw_file=screw_file, dicom_file=dicom_file, cfg_update=cfg_update)

    ann = ds.get_ann(study_id=study_id, projection=projection, body_pos=bodyPos, acquired=acquired,
                     acquired_date=acquired_date, relative_file_path=relative_file_path,
                     units='mm', display=display)  # for maccabi data should use units 'mm' and pixel_space_override (1,1)

    pass


if __name__ == '__main__':

    export_maccabi_to_coco()
    # visualize_single_example()

    pass
