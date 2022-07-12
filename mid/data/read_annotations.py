import pandas as pd
import numpy as np
from pathlib import Path
import json
from mid.data import Annotation

def load_anns_df(anns_dir):

    anns_df = pd.read_parquet(anns_dir)

    # drop duplcates
    main_columns = ['StudyID', 'projection', 'bodyPos', 'acquired']
    ann_id = [id for id in ['vertName', 'rod_id', 'screw_id'] if id in anns_df.columns]
    main_columns += ann_id
    anns_df.drop_duplicates(main_columns, inplace=True)

    return anns_df

def filter_anns_df(df, study_id=None, projection=None, body_pos=None, acquired=None, acquired_date=None, file_id=None):
    """
    Filter annotations data frame.
    At least on of the arguments other than vert_df should be given.
    None value means that filtering does not consider that argument.
    """

    inds = ((study_id is None) or (df.StudyID == study_id)) & \
           ((projection is None) or (df.projection == projection)) & \
           ((body_pos is None) or (df.bodyPos == body_pos)) & \
           ((acquired is None) or (df.acquired == acquired)) & \
           ((acquired_date is None) or (df.acquired_date == acquired_date)) & \
           ((file_id is None) or (df.file_id == file_id))

    df_out = df[inds]

    return df_out


def get_all_vert_anns(df, units='pixel'):
    """
    Get annotations coordinates of all vertebrates.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        Annotation coordinates given as [x, y].
    pixel_spacing : tuple
        Pixel spacing given as (pixel_spacing_x, pixel_spacing_y)
    """

    if len(df) == 0:
        return {}, None, {}

    verts_list = df.vertName.unique()

    anns_dict = {}
    for vert_name in verts_list:
        ann = get_single_vert_anns(df, vert_name, units)
        anns_dict[vert_name] = ann

    # sort alphabetically - for better display
    anns_dict = dict(sorted(anns_dict.items(), key=lambda x: x[0].upper()))  # not really needed

    # get pixel spacing
    # verify that there is only 1 value of pixel spacing for each coordinate
    assert (len(df.pixel_space_x.unique()) == 1) and (len(df.pixel_space_y.unique()) == 1), \
            'currently handling only 1 unique value of pixel spacing, got pixel_spacing_x = {}, pixel_spacing_y = {}' \
            ''.format(df.pixel_space_x.unique(), df.pixel_space_y.unique())
    pixel_spacing = (df.pixel_space_x.iloc[0], df.pixel_space_y.iloc[0])

    metadata = {'dcm_date': str(df.iloc[0].dcm_date) if 'dcm_data' in df.columns else None,
                'relative_file_path': df.iloc[0].relative_file_path if 'relative_file_path' in df.columns else None,
                }

    return anns_dict, pixel_spacing, metadata

def get_single_vert_anns(df, vert_name, units='pixel'):
    """
    Get annotations coordinates of a single vertebrae.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    vert_name : str
        Wanted vertebrae name.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    """

    df = df[df.vertName == vert_name]

    anns_colums = ['lowerVertEnd_x', 'lowerVertEnd_y', 'lowerVertSt_x', 'lowerVertSt_y',
                   'upperVertEnd_x', 'upperVertEnd_y', 'upperVertSt_x', 'upperVertSt_y'
                   ]

    # df = hack_fix_scientific_pixel_spacing(df)  # FIXME: hack for maccabi data, should fix earlier in the data pipe and remove from here in the future

    anns = df[anns_colums].values  # [x, y, x, y, ...] ; corners are given im mm

    anns = anns.reshape(-1, 2)  # [x, y]

    if units == 'pixel':  # convert values from mm to pixels
        pixel_space_x = df.pixel_space_x.values  # [mm / pix]
        pixel_space_y = df.pixel_space_y.values

        # convert values from mm to pixels
        anns[:, 0] /= pixel_space_x  # x
        anns[:, 1] /= pixel_space_y  # y

    return anns

def hack_fix_scientific_pixel_spacing(df):
    """ hack for fixing scientific notation pixel spacing
    """

    path_to_fix_file = '/Users/shilem2/data/maccabi/scientific_notation_paths.json'
    path_to_fix_list = json.loads(Path(path_to_fix_file).read_text())
    path_to_fix_list = [Path(path).parent.as_posix() for path in path_to_fix_list]  # remove file name, leave only relative path to dir

    # get list of verts need fix
    df_vert_to_fix = df.loc[df['relative_file_path'].isin(path_to_fix_list)]

    if len(df_vert_to_fix) == 0:
        return df
    else:
        df = df_vert_to_fix.copy()
        temp_pixel_spacing_xy = (df.pixel_space_x.iloc[0], df.pixel_space_y.iloc[0])
        for prefix in ['lowerVert', 'upperVert']:
            for n, axis in enumerate(['x', 'y']):
                st = f'{prefix}St_{axis}'
                end = f'{prefix}End_{axis}'
                df[st] /= temp_pixel_spacing_xy[n]
                df[end] /= temp_pixel_spacing_xy[n]
                df[end] = ((df[end] - df[st]) * temp_pixel_spacing_xy[n] + df[st]).iloc[0]
                df[st] *= temp_pixel_spacing_xy[n]
                df[end] *= temp_pixel_spacing_xy[n]

    return df

def get_all_screw_anns(df, units='pixel'):
    """
    Get annotations coordinates of all screws.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    pixel_spacing : tuple
        Pixel spacing given as (pixel_spacing_x, pixel_spacing_y)
    """

    if len(df) == 0:
        return {}, None, {}

    df = df.reset_index()  # in order to use index as unique id
    id_list = df.index.unique()
    prefix = 'sc'

    anns_dict = {}
    ids_dict = {}
    for n, id in enumerate(id_list):
        ann, name, ids = get_single_screw_anns(df, id, units)
        anns_dict['{}{}_{}'.format(prefix, n, name)] = ann  # create increasing key instead of rod id
        ids_dict['{}{}_{}'.format(prefix, n, name)] = ids

    # get pixel spacing
    # verify that there is only 1 value of pixel spacing for each coordinate
    assert (len(df.pixel_space_x.unique()) == 1) and (len(df.pixel_space_y.unique()) == 1), \
        'currently handling only 1 unique value of pixel spacing, got pixel_spacing_x = {}, pixel_spacing_y = {}' \
        ''.format(df.pixel_space_x.unique(), df.pixel_space_y.unique())
    pixel_spacing = (df.pixel_space_x.iloc[0], df.pixel_space_y.iloc[0])

    metadata = {'screw_id_dict': ids_dict}

    return anns_dict, pixel_spacing, metadata

def get_single_screw_anns(df, id, units='pixel'):
    """
    Get annotations coordinates of a single screw.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    """

    df = df[df.index == id]

    id_col = ['screw_id']
    ids = df[id_col].values[0][0]

    anns_colums = ['screwSt_x', 'screwSt_y', 'screwEnd_x', 'screwEnd_y']
    anns = df[anns_colums].values  # [x, y, x, y, ...] ; corners are given im mm
    anns = anns.reshape(-1, 2)  # [x, y]
    name = df['vertName'].values[0] if 'vertName' in df.columns else 'sc'

    if units == 'pixel':  # convert values from mm to pixels
        pixel_space_x = df.pixel_space_x.values  # [mm / pix]
        pixel_space_y = df.pixel_space_y.values

        # convert values from mm to pixels
        anns[:, 0] /= pixel_space_x  # x
        anns[:, 1] /= pixel_space_y  # y

    return anns, name, ids

def get_all_rod_anns(df, units='pixel'):
    """
    Get annotations coordinates of all rods.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    pixel_spacing : tuple
        Pixel spacing given as (pixel_spacing_x, pixel_spacing_y)
    """

    if len(df) == 0:
        return {}, None, {}

    df = df.reset_index()  # in order to use index as unique id
    id_list = df.index.unique()
    prefix = 'r'

    anns_dict = {}
    ids_dict = {}
    for n, id in enumerate(id_list):
        ann, name, ids = get_single_rod_anns(df, id, units)
        anns_dict['{}{}_{}'.format(prefix, n, name)] = ann  # create increasing key instead of rod id
        ids_dict['{}{}_{}'.format(prefix, n, name)] = ids

    # get pixel spacing
    # verify that there is only 1 value of pixel spacing for each coordinate
    assert (len(df.pixel_space_x.unique()) == 1) and (len(df.pixel_space_y.unique()) == 1), \
        'currently handling only 1 unique value of pixel spacing, got pixel_spacing_x = {}, pixel_spacing_y = {}' \
        ''.format(df.pixel_space_x.unique(), df.pixel_space_y.unique())
    pixel_spacing = (df.pixel_space_x.iloc[0], df.pixel_space_y.iloc[0])

    # get metadata
    metadata = {'rod_id_dict': ids_dict}

    return anns_dict, pixel_spacing, metadata

def get_single_rod_anns(df, id, units='pixel'):
    """
    Get annotations coordinates of a single rod.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    """

    df = df[df.index == id]

    id_col = ['rod_id']
    ids = df[id_col].values[0][0]

    anns_colums = ['rodSt_x', 'rodSt_y', 'rodEnd_x', 'rodEnd_y']
    anns = df[anns_colums].values  # [x, y, x, y, ...] ; corners are given im mm
    anns = anns.reshape(-1, 2)  # [x, y]
    name = df['vertName'].values[0] if 'vertName' in df.columns else 'r'

    if units == 'pixel':  # convert values from mm to pixels
        pixel_space_x = df.pixel_space_x.values  # [mm / pix]
        pixel_space_y = df.pixel_space_y.values

        # convert values from mm to pixels
        anns[:, 0] /= pixel_space_x  # x
        anns[:, 1] /= pixel_space_y  # y

    return anns, name, ids

def get_all_icl_anns(df, units='pixel'):
    """
    Get annotations coordinates of all icl.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    pixel_spacing : tuple
        Pixel spacing given as (pixel_spacing_x, pixel_spacing_y)
    """

    if len(df) == 0:
        return {}, None

    df = df.reset_index()  # in order to use index as unique id
    id_list = df.index.unique()
    prefix = 'icl'

    anns_dict = {}
    for n, id in enumerate(id_list):
        ann = get_single_icl_anns(df, id, units)
        anns_dict['{}{}'.format(prefix, n)] = ann  # create increasing key instead of rod id

    # get pixel spacing
    # verify that there is only 1 value of pixel spacing for each coordinate
    assert (len(df.pixel_space_x.unique()) == 1) and (len(df.pixel_space_y.unique()) == 1), \
        'currently handling only 1 unique value of pixel spacing, got pixel_spacing_x = {}, pixel_spacing_y = {}' \
        ''.format(df.pixel_space_x.unique(), df.pixel_space_y.unique())
    pixel_spacing = (df.pixel_space_x.iloc[0], df.pixel_space_y.iloc[0])

    return anns_dict, pixel_spacing

def get_single_icl_anns(df, id, units='pixel'):
    """
    Get annotations coordinates of a single icl.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    """

    df = df[df.index == id]
    anns_colums = ['ICLSt_x', 'ICLSt_y', 'ICLEnd_x', 'ICLEnd_y']

    anns = df[anns_colums].values  # [x, y, x, y, ...] ; corners are given im mm
    anns = anns.reshape(-1, 2)  # [x, y]

    if units == 'pixel':  # convert values from mm to pixels
        pixel_space_x = df.pixel_space_x.values  # [mm / pix]
        pixel_space_y = df.pixel_space_y.values

        # convert values from mm to pixels
        anns[:, 0] /= pixel_space_x  # x
        anns[:, 1] /= pixel_space_y  # y

    return anns

def get_all_femur_anns(df, units='pixel'):
    """
    Get annotations coordinates of all femur.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    pixel_spacing : tuple
        Pixel spacing given as (pixel_spacing_x, pixel_spacing_y)
    """

    if len(df) == 0:
        return {}, None

    df = df.reset_index()  # in order to use index as unique id
    id_list = df.index.unique()
    prefix = 'f'

    anns_dict = {}
    for n, id in enumerate(id_list):
        ann = get_single_femur_anns(df, id, units)
        anns_dict['{}{}'.format(prefix, n)] = ann  # create increasing key instead of rod id

    # get pixel spacing
    # verify that there is only 1 value of pixel spacing for each coordinate
    assert (len(df.pixel_space_x.unique()) == 1) and (len(df.pixel_space_y.unique()) == 1), \
        'currently handling only 1 unique value of pixel spacing, got pixel_spacing_x = {}, pixel_spacing_y = {}' \
        ''.format(df.pixel_space_x.unique(), df.pixel_space_y.unique())
    pixel_spacing = (df.pixel_space_x.iloc[0], df.pixel_space_y.iloc[0])

    return anns_dict, pixel_spacing

def get_single_femur_anns(df, id, units='pixel'):
    """
    Get annotations coordinates of a single femur.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations data frame.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.

    Returns
    -------
    anns : ndarray
        annotation coordinates given as [x, y].
    """

    df = df[df.index == id]
    anns_colums = ['femurSt_x', 'femurSt_y', 'femurEnd_x', 'femurEnd_y']

    anns = df[anns_colums].values  # [x, y, x, y, ...] ; corners are given im mm
    anns = anns.reshape(-1, 2)  # [x, y]

    if units == 'pixel':  # convert values from mm to pixels
        pixel_space_x = df.pixel_space_x.values  # [mm / pix]
        pixel_space_y = df.pixel_space_y.values

        # convert values from mm to pixels
        anns[:, 0] /= pixel_space_x  # x
        anns[:, 1] /= pixel_space_y  # y

    return anns


def get_scan_anns(vert_df=None, rod_df=None, screw_df=None, dicom_df=None, icl_df=None, femur_df=None,
                  study_id=None, projection=None, body_pos=None, acquired=None, acquired_date=None, file_id=None, units='mm', pixel_spacing_override=None, display=False, save_fig_name=None):
    """
    Get scan annotations.

    Parameters
    ----------
    vert_df : pandas.DataFrame
        Vertebrates annotations.
    rod_df : pandas.DataFrame
        Rods annotations.
    screw_df : pandas.DataFrame
        Screws annotations.
    dicom_df : pandas.DataFrame
        Paths to Dicom files.
    icl_df : pandas.DataFrame
        Icl annotations.
    femur_df : pandas.DataFrame
        Femur annotations.
    study_id, projection, body_pos, acquired, file_id : str, optional
        Scan parameters. At least on of the arguments other than vert_df should be given.
        None value means that filtering does not consider that argument.
    units : str, optional
        Corner coordinates value type, one of {'pixel' , 'mm'}
        Original annotations value type is 'mm'.
    display : bool, optional
        If True, annotations will be displayed on top of the image. dicom_df must be given.

    Returns
    -------
    ann : spireg.data.annotation.Annotation
     Annotation object.
    """
    if (vert_df is None) and (rod_df is None) and (screw_df is None) and (dicom_df is None):
        raise ValueError('At least one if input dataframes should be not None!')

    anns_vert = {}
    metadata_vert = {}
    anns_rod = {}
    metadata_rod = {}
    anns_screw = {}
    metadata_screw = {}
    anns_icl = {}
    anns_femur = {}
    pixel_spacing_vert = pixel_spacing_rod = pixel_spacing_screw = pixel_spacing_icl = pixel_spacing_femur = None
    dicom_path = None

    if vert_df is not None:
        vert_df = filter_anns_df(vert_df, study_id, projection, body_pos, acquired, acquired_date, file_id)
        anns_vert, pixel_spacing_vert, metadata_vert = get_all_vert_anns(vert_df, units)

    if rod_df is not None:
        rod_df = filter_anns_df(rod_df, study_id, projection, body_pos, acquired, acquired_date, file_id)
        anns_rod, pixel_spacing_rod, metadata_rod = get_all_rod_anns(rod_df, units)

    if screw_df is not None:
        screw_df = filter_anns_df(screw_df, study_id, projection, body_pos, acquired, acquired_date, file_id)
        anns_screw, pixel_spacing_screw, metadata_screw = get_all_screw_anns(screw_df, units)

    if icl_df is not None:
        icl_df = filter_anns_df(icl_df, study_id, projection, body_pos, acquired, acquired_date, file_id)
        anns_icl, pixel_spacing_icl = get_all_icl_anns(icl_df, units)

    if femur_df is not None:
        femur_df = filter_anns_df(femur_df, study_id, projection, body_pos, acquired, acquired_date, file_id)
        anns_femur, pixel_spacing_femur = get_all_femur_anns(femur_df, units)

    if dicom_df is not None:
        dicom_df = filter_anns_df(dicom_df, study_id, projection, body_pos, acquired, acquired_date, file_id)
        assert len(dicom_df) == 1, 'only 1 dicom path should remain after filtering'
        dicom_path = dicom_df['dicom_path'].values[0]

    # merge anns
    ann_dict = {**anns_vert, **anns_rod, **anns_screw, **anns_icl, **anns_femur}
    metadata_df = {**metadata_vert, **metadata_rod, **metadata_screw}

    # merge pixel spacing
    assert verify_equal_or_None(pixel_spacing_vert, pixel_spacing_screw, pixel_spacing_rod, pixel_spacing_icl, pixel_spacing_femur), 'currenlty only one value of pixel spacing for all elements is supported'
    pixel_spacing = pixel_spacing_vert

    pixel_spacing_orig = np.array(pixel_spacing)
    if pixel_spacing_override is not None:
        pixel_spacing = pixel_spacing_override

    # pack all data in Annotation object
    metadata = {'study_id': study_id, 'projection': projection, 'body_pos': body_pos, 'acquired': acquired, 'acquired_date': acquired_date, 'file_id': file_id, 'pixel_spacing_orig': pixel_spacing_orig, **metadata_df}
    ann = Annotation(ann_dict, pixel_spacing, units, dicom_path, metadata, display=display, save_fig_name=save_fig_name)

    return ann

def verify_equal_or_None(*data):
    """Test if all elements in data are equal to each other or None.
    """
    if len(data) <= 1:
        return True

    d0 = data[0]
    for d in data:
        if d is not None:
            if d != d0:
                return False

    return True

def get_dicom_paths(root_dir, json_pattern = '**/XR/**/*det.json', dicom_pattern='*.dcm'):

    # find jsons
    paths = Path(root_dir).rglob(json_pattern)
    # paths_list = sorted(list(paths))  # takes too much time!!!

    # get dicom path
    wanted_keys = ['StudyID', 'acquired', 'bodyPos', 'projection']
    srs = []
    for path in paths:  #paths_list:
        text = path.read_text()
        data_dict = json.loads(text)
        data_wanted = {key: data_dict[key] for key in wanted_keys}
        dicom_path = list(path.parent.glob(dicom_pattern))
        if len(dicom_path) != 1:  # assume that each folder contains exactly 1 dicom file
            raise ValueError('Only 1 dcm file should be in each folder!')
        data_wanted['dicom_path'] = dicom_path[0].resolve().as_posix()
        sr = pd.Series(data_wanted)
        srs.append(sr)

    df = pd.concat(srs, axis=1).transpose()

    return df
