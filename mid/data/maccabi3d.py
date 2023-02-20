import pandas as pd

from mid.data import Dataset
# from .read_3d_data import
from mid.config import config


class Maccbi3dDataset(Dataset):
    """Dataset class, enables to get annotations, filter data, etc.
    """

    def __init__(self,
                 meta_file,
                 cfg_update=None,
                 ):

        if cfg_update is None:
            cfg_update = dict()

        cfg_update.update({'meta_file': meta_file,
                           })

        # update config
        cfg_default = config.load_default_config('data3d')
        cfg = config.merge_config(cfg_default, cfg_update)

        # read data frames
        meta_df = pd.read_parquet(meta_file)

        dataset = {'meta_df': meta_df,
                   }

        self.cfg = cfg  # config is saved to file inside RegisterByKeypoints.__init__()
        self.dataset = dataset

        pass

    def get_study_id_list(self, key='meta_df', col_name='study_id'):
        s = sorted(self.dataset[key][col_name].unique())
        return s

    def get_ann(self):

        pass

    def filter_study_id(self, study_id, key='meta_df', is_preop=None, is_postop=None):
        df = Maccbi3dDataset.filter_anns_df(self.dataset[key], study_id=study_id, is_preop=is_preop, is_postop=is_postop)
        return df

    @staticmethod
    def filter_anns_df(df, study_id=None, mongo_id=None, relative_file_path=None, dcm_date=None, surgery_date=None,
                       is_preop=None, is_postop=None):
        """
        Filter annotations data frame.
        At least on of the arguments other than vert_df should be given.
        None value means that filtering does not consider that argument.
        """

        inds = ((study_id is None) or (df.study_id == study_id)) & \
               ((mongo_id is None) or (df.mongo_id == mongo_id)) & \
               ((relative_file_path is None) or (df.relative_file_path == relative_file_path)) & \
               ((dcm_date is None) or (df.dcm_date == dcm_date)) & \
               ((surgery_date is None) or (df.surgery_date == surgery_date)) & \
               ((is_preop is None) or (df.is_preop == is_preop)) & \
               ((is_postop is None) or (df.is_postop == is_postop))

        df_out = df[inds]

        return df_out



