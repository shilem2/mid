import pandas as pd

from mid.data import Dataset
from .read_annotations import get_scan_anns, filter_anns_df
from mid.config import config


class MaccbiDataset(Dataset):
    """Dataset class, enables to get annotations, filter data, etc.
    """

    def __init__(self,
                 cfg_update=None,
                 vert_file=None,
                 pair_file=None,
                 screw_file=None,
                 rod_file=None,
                 icl_file=None,
                 femur_file=None,
                 dicom_file=None,
                 ):

        #
        if cfg_update is None:
            cfg_update = dict()

        cfg_update.update({'vert_file': vert_file,
                           'pair_file': pair_file,
                           'screw_file': screw_file,
                           'rod_file': rod_file,
                           'icl_file': icl_file,
                           'femur_file': femur_file,
                           'dicom_file': dicom_file
                           })

        # update config
        cfg_default = config.load_default_config('data')
        cfg = config.merge_config(cfg_default, cfg_update)

        # read data frames
        vert_df = pd.read_parquet(vert_file)
        pair_df = pd.read_parquet(pair_file) if pair_file is not None else None
        screw_df = pd.read_parquet(screw_file) if screw_file is not None else None
        rod_df = pd.read_parquet(rod_file) if rod_file is not None else None
        icl_df = pd.read_parquet(icl_file) if icl_file is not None else None
        femur_df = pd.read_parquet(femur_file) if femur_file is not None else None
        dicom_df = pd.read_parquet(dicom_file) if dicom_file is not None else None

        dataset = {'vert_df': vert_df,
                   'pair_df': pair_df,
                   'rod_df': rod_df,
                   'screw_df': screw_df,
                   'icl_df': icl_df,
                   'femur_df': femur_df,
                   'dicom_df': dicom_df,
                   }

        self.cfg = cfg  # config is saved to file inside RegisterByKeypoints.__init__()
        self.dataset = dataset

        # members
        self.acquired_dates = ['PreOp', 'PostOp', 'Week 3-6', 'Month 6', 'Month 12']  # used for sort

        pass

    def get_study_id_list(self, key='vert_df', col_name='StudyID'):
        s = sorted(self.dataset[key][col_name].unique())
        return s

    def get_file_id_list(self, key='vert_df', col_name='file_id'):
        s = sorted(self.dataset[key][col_name].unique())
        return s

    def get_ann(self, study_id=None, projection=None, body_pos=None, acquired=None, acquired_date=None, file_id=None, relative_file_path=None, units='mm', display=False, save_fig_name=None):
        ann = get_scan_anns(self.dataset['vert_df'], self.dataset['rod_df'], self.dataset['screw_df'], self.dataset['dicom_df'], self.dataset['icl_df'], self.dataset['femur_df'],
                            study_id, projection, body_pos, acquired, acquired_date, file_id, relative_file_path, units, self.cfg['pixel_spacing_override'], display, save_fig_name)
        return ann

    def filter_study_id(self, study_id, key='vert_df'):
        df = filter_anns_df(self.dataset[key], study_id=study_id)
        return df

    def sort_acquired_dates(self, acquired_dates):
        """Sort acquired_dates chronologically.
        """
        indices_ordered = list(range(len(self.acquired_dates)))
        zipped_sorted_ind_vert = list(zip(indices_ordered, self.acquired_dates))
        indices = sorted([ind for (ind, acquired_date) in zipped_sorted_ind_vert if acquired_date in acquired_dates])  # indices of input keys
        acquired_dates_ordered = [self.acquired_dates[ind] for ind in indices]

        return acquired_dates_ordered


    def find_pairs_for_registration(self, study_id):

        if self.cfg['pairs_for_registration']['acquired_date'] == 'same':
            study_pairs_df = self.find_pairs_same_acquired_date(study_id, self.cfg['pairs_for_registration']['skip_flipped_anns'])
        elif self.cfg['pairs_for_registration']['acquired_date'] == 'different':
            study_pairs_df = self.find_pairs_different_acquired_date(study_id, latest_preop=self.cfg['pairs_for_registration']['latest_preop'], skip_flipped_anns=self.cfg['pairs_for_registration']['skip_flipped_anns'])
        return study_pairs_df

    def find_pairs_same_acquired_date(self, study_id, skip_flipped_anns=False):
        """Find pairs of same projection, body pose and acquired_date, acquired at different times for a specific study_id.
        """

        study_df = self.filter_study_id(study_id)
        df = study_df.drop_duplicates('file_id')
        if skip_flipped_anns:
            inds = df.x_sign != -1
            df = df[inds]
        df = df[['file_id', 'StudyID', 'projection', 'acquired_date', 'bodyPos', 'acquired']]  # use only most important columns

        acquired_date_list = self.get_unique_val_list(df, 'acquired_date')

        # get pairs of different scans acquired at same acquired_date (e.g. several PreOp scans, several Month6 scans, etc.)
        study_df_list = []
        for acquired_date in acquired_date_list:
            acquired_date_df = self.filter_anns_df(df, acquired_date=acquired_date)
            df_pairs = acquired_date_df.merge(acquired_date_df.copy(), on=['StudyID', 'projection', 'bodyPos'], how='inner')  # Cartesian product
            df_pairs = df_pairs[df_pairs.file_id_x < df_pairs.file_id_y]  # filter 1) duplicated entries with swapped file ids 2) identical file_id
            study_df_list.append(df_pairs)

        study_pairs_df = pd.concat(study_df_list)

        return study_pairs_df

    def find_pairs_different_acquired_date(self, study_id, latest_preop=True, preop_must=True, skip_flipped_anns=False):
        """Find pairs of same projection and body pose, and different acquired_date, for a specific study_id.
        """

        study_df = self.filter_study_id(study_id, key='dicom_df')
        df = study_df.drop_duplicates('file_id')
        if skip_flipped_anns:
            inds = df.x_sign != -1
            df = df[inds]
        if latest_preop:
            df = keep_latest_preop(df, groupCols=['StudyID', 'acquired_date', 'projection', 'bodyPos'], sortCols=['StudyID', 'dcm_date', 'projection', 'bodyPos'], verbose=False)
        df = df[['file_id', 'StudyID', 'projection', 'acquired_date', 'bodyPos', 'acquired', 'relative_file_path', 'dicom_path', 'x_sign']]  # use only most important columns

        acquired_date_list = self.sort_acquired_dates(self.get_unique_val_list(df, 'acquired_date'))

        if preop_must and ('PreOp' not in acquired_date_list):
            study_pairs_df = pd.DataFrame()  # empty df
        else:
            # get pairs of scans acquired at different acquired_date
            acquired_date_ref = acquired_date_list[0]  # use earliest acquired date as reference
            acquired_date_ref_df = self.filter_anns_df(df, acquired_date=acquired_date_ref, equals=True)  # find all entries from current acquired date
            acquired_date_other_df = self.filter_anns_df(df, acquired_date=acquired_date_ref, equals=False)  # find all entries from other acquired dates
            study_pairs_df = acquired_date_ref_df.merge(acquired_date_other_df.copy(),
                                                        on=['StudyID', 'projection', 'bodyPos'],
                                                        how='outer')

        return study_pairs_df


    @staticmethod
    def get_unique_val_list(df, col_name):
        s = sorted(df[col_name].unique())
        return s

    @staticmethod
    def filter_anns_df(df, study_id=None, projection=None, body_pos=None, acquired=None, acquired_date=None,
                       dicom_path=None, relative_file_path=None, file_id=None, equals=True):
        """
        Filter annotations data frame.
        At least on of the arguments other than df should be given.
        None value means that filtering does not consider that argument.
        """

        if equals:
            inds = ((study_id is None) | (df.StudyID == study_id)) & \
                   ((projection is None) | (df.projection == projection)) & \
                   ((body_pos is None) | (df.bodyPos == body_pos)) & \
                   ((acquired is None) | (df.acquired == acquired)) & \
                   ((acquired_date is None) | (df.acquired_date == acquired_date)) & \
                   ((dicom_path is None) | (df.dicom_path == dicom_path)) & \
                   ((relative_file_path is None) | (df.relative_file_path == relative_file_path)) & \
                   ((file_id is None) | (df.file_id == file_id))
        else:
            inds = ((study_id is None) | (df.StudyID != study_id)) & \
                   ((projection is None) | (df.projection != projection)) & \
                   ((body_pos is None) | (df.bodyPos != body_pos)) & \
                   ((acquired is None) | (df.acquired != acquired)) & \
                   ((acquired_date is None) | (df.acquired_date != acquired_date)) & \
                   ((dicom_path is None) | (df.dicom_path != dicom_path)) & \
                   ((relative_file_path is None) | (df.relative_file_path != relative_file_path)) & \
                   ((file_id is None) | (df.file_id != file_id))

        df_out = df[inds]

        return df_out

# # code from Dan: giraffe/complication_prediction/preprocessing/ALD_target.ipynb

def keep_latest_preop(df, groupCols=['StudyID', 'acquired_date', 'projection', 'bodyPos'], sortCols=['StudyID', 'dcm_date', 'projection', 'bodyPos'], verbose=False):
    """
    Keeps last preop row (and all other/postop rows), according to dcm_date
    """

    mask = df["acquired_date"] == "PreOp"
    df["dcm_date"] = pd.to_datetime(df["dcm_date"], infer_datetime_format=True)
    df_pre = df[mask].copy()

    if verbose:
        print("before dropping duplicate pre")
        count_df_ids(df_pre)

    df_post = df[~mask]
    df_pre = df_pre.sort_values(by=["dcm_date"], ascending=False)
    df_pre = df_pre.drop_duplicates(subset=groupCols, keep="first")

    if verbose:
        print("after dropping duplicate pre")
        count_df_ids(df_pre)
    df = pd.concat([df_pre, df_post])

    ### I don't remember if this sort / sort order is strictly. necessary! Can test it..
    partial_sort_keys = [i for i in groupCols if i != "StudyID"]  # group keys except for study ID,
    #     df = df.sort_values(by=["StudyID","dcm_date","vertNum_bottom","vertNum_top"],ascending=True)## ORIG - broken if verts not present
    df = df.sort_values(by=sortCols + partial_sort_keys, ascending=True)

    return df


def keep_median_instances(df, target: str = "disc_height_robust",  # _normed
                          groupCols=["StudyID", "acquired_date", "acquired", "vertNum_bottom", "vertNum_top"],
                          filter_latest_preop=True):
    """
    Keep 1 row per "grouped period", according to the row whose `target` value is closest to the median.
    Opt: do Extra filtering for Preop (Keep latest row/period for preops) (uses `keep_latest_preop` function)
    """
    #     count_df_ids(df)
    orig_id_counts = df["StudyID"].nunique()
    df[f"rank_{target}"] = (df.groupby(groupCols)[target].transform("rank", pct=True) - 0.5).abs()
    df.sort_values(by=[f"rank_{target}"], ascending=True, inplace=True)
    df = df.drop_duplicates(subset=groupCols, keep="first")
    df.sort_values(by=groupCols, ascending=True, inplace=True)
    assert orig_id_counts == df["StudyID"].nunique()
    count_df_ids(df)
    if filter_latest_preop:
        df = keep_latest_preop(df)
    return df


# PREV:
#
# def keep_latest_preop(df, groupCols=["StudyID", "vertNum","projection"],
#                       sortCols=["StudyID", "dcm_date", "vertNum"], verbose=False):
#     """
#     Keeps last preop row (and all other/postop rows), according to dcm_date
#     """
#     mask = df["acquired_date"] == "PreOp"
#     # df["dcm_date"] = pd.to_datetime(df["dcm_date"], infer_datetime_format=True)  # may cause error if already is a datetime - if so, disable this or do an if beofre
#     df_pre = df[mask].copy()
#     if verbose:
#         count_df_ids(df_pre)
#     df_post = df[~mask]
#
#     df_pre["dcm_date"] = pd.to_datetime(df_pre["dcm_date"],infer_datetime_format=True)
#     df_pre = df_pre.sort_values(by=["dcm_date"], ascending=False)
#     df_pre = df_pre.drop_duplicates(subset=groupCols, keep="first")
#     if verbose:
#         print("after dropping duplicate pre")
#         count_df_ids(df_pre)
#     df = pd.concat([df_pre, df_post])
#
#     df = df.sort_values(by=sortCols, ascending=True)
#     return df
#
#
# def keep_median_instances(df, target: str = "disc_height_robust",  # _normed
#                           groupCols=["StudyID", "acquired_date", "acquired", "vertNum_bottom", "vertNum_top"],
#                           filter_latest_preop=True,
#                           sortCols=["StudyID", "dcm_date", "vertNum_bottom", "vertNum_top"]):
#     """
#     Keep 1 row per "grouped period", according to the row whose `target` value is closest to the median.
#     Opt: do Extra filtering for Preop (Keep latest row/period for preops) (uses `keep_latest_preop` function)
#     """
#     orig_id_counts = df["StudyID"].nunique()
#     df[f"rank_{target}"] = (df.groupby(groupCols)[target].transform("rank", pct=True) - 0.5).abs()
#     df.sort_values(by=[f"rank_{target}"], ascending=True, inplace=True)  # FIXME: ascending False ??
#     df = df.drop_duplicates(subset=groupCols)
#     df.sort_values(by=groupCols, ascending=True, inplace=True)  # FIXME: ascending False ??
#     assert orig_id_counts == df["StudyID"].nunique()
#     count_df_ids(df)
#     if filter_latest_preop:
#         df = keep_latest_preop(df, groupCols, sortCols)
#     return df

def count_df_ids(df):
    print(df.shape[0],"rows")
    print(df.reset_index().filter(["StudyID","file_id"],axis=1).nunique())





