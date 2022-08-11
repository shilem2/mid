import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd


def generate_maccabi_df():

    rel2full_path_file = '/mnt/magic_efs/moshe/implant_detection/data/2022-08-10_merged_data_v2/dicom/short2full.json'
    dicom_pattern = '*.dcm'
    num_max = -1
    output_df_file = Path(rel2full_path_file).parent / 'dicom.parquet'


    text = Path(rel2full_path_file).read_text()
    dicom_path_dict = json.loads(text)
    dicom_path_dict = {rel_path: Path(full_path.replace('M:', '/media/MazorData').replace('\\', '/')).parent for rel_path, full_path in dicom_path_dict.items()}

    srs = []
    for n, (rel_path, full_path) in tqdm(enumerate(dicom_path_dict.items()), total=len(dicom_path_dict)):

        dicom_path_list = list(full_path.glob(dicom_pattern))
        if len(dicom_path_list) == 0:
            continue
        dicom_path = dicom_path_list[0].as_posix()  # assume only 1 dicom in each


        path_dict = {
            'relative_file_path': rel_path,
            'full_file_path': dicom_path
        }

        sr = pd.Series(path_dict)
        srs.append(sr)

        if (num_max > 0) and (n >= (num_max - 1)):
            break

    df = pd.concat(srs, axis=1).transpose()

    df.to_parquet(output_df_file)

    pass



if __name__ == '__main__':

    generate_maccabi_df()

    pass