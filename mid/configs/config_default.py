from pathlib import Path

data = {'pixel_spacing_override': None,
        'vert_file': None,
        'pair_file': None,
        'screw_file': None,
        'rod_file': None,
        'icl_file': None,
        'femur_file': None,
        'dicom_file': None,
        'pairs_for_registration': {'acquired_date': 'different',
                                   'skip_flipped_anns': False,
                                   'latest_preop': True,
                                   'latest_postop': False,
                                   'projection': None,
                                   'body_pose': None,
                                   }
        }

data3d = {'meta_file': None,
          }

cfg = {'data': data,
       'data3d': data3d,
       }
