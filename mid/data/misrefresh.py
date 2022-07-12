from .read_annotations import load_anns_df, get_scan_anns, get_dicom_paths
from mid.data import Dataset


class MisRefreshDataset(Dataset):
    """Dataset class, enables to get annotations, filter data, etc.
    """

    def __init__(self,
                 vert_dir='/Users/shilem2/data/mis_refresh/mis_refresh_2d_features_new_asc/vert_data/',
                 rod_dir='/Users/shilem2/data/mis_refresh/mis_refresh_2d_features_new_asc/rod_data/',
                 screw_dir='/Users/shilem2/data/mis_refresh/mis_refresh_2d_features_new_asc/screw_data/',
                 dicom_dir='/Users/shilem2/Library/CloudStorage/OneDrive-SharedLibraries-MedtronicPLC/Lev-Tov, Amir - Spine team data/Data/mis-refresh-raw-xr/',
                 ):

        vert_df = load_anns_df(vert_dir)
        rod_df = load_anns_df(rod_dir)
        screw_df = load_anns_df(screw_dir)
        dicom_df = get_dicom_paths(dicom_dir)

        dataset = {'vert_df': vert_df,
                   'rod_df': rod_df,
                   'screw_df': screw_df,
                   'dicom_df': dicom_df
                   }

        self.dataset = dataset

    def get_study_id_list(self, key='dicom_df', col_name='StudyID'):
        s = sorted(self.dataset[key][col_name].unique())
        return s

    def get_ann(self, study_id=None, projection=None, body_pos=None, acquired=None, units='mm', display=False):
        ann = get_scan_anns(self.dataset['vert_df'], self.dataset['rod_df'], self.dataset['screw_df'], self.dataset['dicom_df'], None, None,
                            study_id, projection, body_pos, acquired, units=units, display=display)
        return ann


    def find_pairs(self):
        """Find pairs of same projection and body pose, acquired at different times for all study_ids in dataset.
        """
        pass

    def find_pairs_different_acquired_time(self, study_id, filter_preop=True, filter_postop=False):
        """Given a study_id, find pairs of same projection and body pose acquired at different times.
        """

        # filter by study_id
        study_df = self.filter_anns_df(self.dataset['dicom_df'], study_id=study_id)

        if filter_preop:
            study_df = study_df[study_df.acquired != 'PreOp']

        if filter_postop:
            study_df = study_df[study_df.acquired == 'PreOp']

        # find pairs of different acquired times
        df = study_df.merge(study_df.copy(), on=['StudyID', 'projection', 'bodyPos'], how='inner')  # Cartesian product
        df = df[df.acquired_x != df.acquired_y]  # filter identical acquired time

        return df

    @staticmethod
    def filter_anns_df(df, study_id=None, projection=None, body_pos=None, acquired=None, dicom_path=None, file_id=None):
        """
        Filter annotations data frame.
        At least on of the arguments other than vert_df should be given.
        None value means that filtering does not consider that argument.
        """

        inds = ((study_id is None) | (df.StudyID == study_id)) & \
               ((projection is None) | (df.projection == projection)) & \
               ((body_pos is None) | (df.bodyPos == body_pos)) & \
               ((acquired is None) | (df.acquired == acquired)) & \
               (('dicom_path' not in df.columns) or (dicom_path is None) | (df.dicom_path == dicom_path)) & \
               (('file_id' not in df.columns) or (file_id is None) | (df.file_id == file_id))

        df_out = df[inds]

        return df_out




