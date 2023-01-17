import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def generate_maccabi_dicom_df():

    # rel2full_path_file = '/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/dicom/short2full.json'
    rel2full_path_file = '/mnt/magic_efs/moshe/implant_detection/data/2023-08-10_merged_data_v2/dicom/short2full_dcm.json'
    # rel2full_path_file = '/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/dicom/short2full_dcm.json'
    dicom_pattern = '*.dcm'
    num_max = -1
    output_df_file = Path(rel2full_path_file).parent / 'dicom_path_rel2full.parquet'

    text = Path(rel2full_path_file).read_text()
    dicom_path_dict = json.loads(text)
    dicom_path_dict = {rel_path.lstrip('/'): Path(full_path.replace('M:', '/media/MazorData').replace('\\', '/')).parent for rel_path, full_path in dicom_path_dict.items()}

    srs = []
    for n, (rel_path, full_path) in tqdm(enumerate(dicom_path_dict.items()), total=len(dicom_path_dict)):

        dicom_path_list = list(full_path.glob(dicom_pattern))
        if len(dicom_path_list) == 0:
            continue
        dicom_path = dicom_path_list[0].as_posix()  # assume only 1 dicom in each


        path_dict = {
            'relative_file_path': rel_path,
            'dicom_path': dicom_path
        }

        sr = pd.Series(path_dict)
        srs.append(sr)

        if (num_max > 0) and (n >= (num_max - 1)):
            break

    df = pd.concat(srs, axis=1).transpose()

    df.to_parquet(output_df_file)

    pass

def add_fields_maccabi_dicom_df():

    # dicom_path = '/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/dicom/dicom_path_rel2full.parquet'
    # vert_path = '/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/vert/vert.parquet'
    dicom_path = '/mnt/magic_efs/moshe/implant_detection/data/2023-08-10_merged_data_v2/dicom/dicom_path_rel2full.parquet'
    vert_path = '/mnt/magic_efs/moshe/implant_detection/data/2023-08-10_merged_data_v2/vert/vert.parquet'

    dicom_df = pd.read_parquet(dicom_path)
    vert_df = pd.read_parquet(vert_path)

    wanted_columns = ['StudyID', 'acquired', 'acquired_date', 'projection', 'bodyPos', 'file_id', 'relative_file_path', 'dcm_date', 'x_sign']
    df = dicom_df.merge(vert_df[wanted_columns], on=['relative_file_path'])
    df = df.drop_duplicates(['relative_file_path'])

    dicom_df_out_path = Path(dicom_path).parent / 'dicom.parquet'
    df.to_parquet(dicom_df_out_path)  # override input dicom file

    pass




if __name__ == '__main__':

    generate_maccabi_dicom_df()
    add_fields_maccabi_dicom_df()

    pass