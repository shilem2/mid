import numpy as np
import pytest
from pytest import approx
from skimage.transform import SimilarityTransform

from mid.data import Annotation
from mid.tests import read_test_data, read_data

def test_create_annotation_instance():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('Maccabi_19359.dat')[0:6]

    ann = Annotation(ann1.ann, pixel_spacing, units)

    assert ann.units == 'pixel'
    assert ann.pixel_spacing == approx(np.array([1., 1.]))
    assert str(ann) == 'Annotation(num_elements=7, pixel_spacing=[1. 1.], units=pixel)'
    assert ann['L1'] == approx(np.array([[647.815  , 290.822  ],
                                         [878.881  , 323.7665 ],
                                         [892.8575 , 460.59105],
                                         [641.226  , 454.255  ]]))
    assert set(ann) == set(ann1)
    keys = [a for a in ann]  # iterable
    ann_dict = {k: v for k, v in ann.items()}  # dict like

    # test wrong initializations
    with pytest.raises(AssertionError):
        Annotation(np.array(ann1), [0.1], 'cm')  # wrong ann type - must be dict
    with pytest.raises(AssertionError):
        Annotation(ann1, [0.1], 'cm')  # wrong units
    with pytest.raises(AssertionError):
        Annotation(ann1, [0.1], units)  # single pixel_spacing

    pass

def test_get_values():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('MR04-019.dat')[0:6]

    ann = Annotation(ann1, pixel_spacing, units)

    values_xy_pixel1 = ann.values(order='xy', units=None)
    values_xy_pixel2 = ann.values(order='xy')
    values_xy_pixel3 = ann.values(order='xy', units='pixel')
    values_rc_pixel = ann.values(order='rc', units='pixel')
    values_xy_mm = ann.values(order='xy', units='mm')
    values_rc_mm = ann.values(order='rc', units='mm')

    # reference values
    L1_mm = np.array([[219.9, 128.8],
                      [188.3, 138.7],
                      [206.1, 101.9],
                      [171.3, 112.6]])
    L1_pixel = L1_mm / pixel_spacing

    assert approx(L1_pixel) == values_xy_pixel1[:4, :]
    assert approx(L1_pixel) == values_xy_pixel2[:4, :]
    assert approx(L1_pixel) == values_xy_pixel3[:4, :]
    assert approx(L1_pixel[:, [1, 0]]) == values_rc_pixel[:4, :]
    assert approx(L1_mm) == values_xy_mm[:4, :]
    assert approx(L1_mm[:, [1, 0]]) == values_rc_mm[:4, :]

    pass

def test_change_units():

    ann1, ann2, img1, img2, pixel_spacing, units = read_test_data('MR04-019.dat')[0:6]

    ann = Annotation(ann1, pixel_spacing, units)

    # reference values
    L1_mm = np.array([[219.9, 128.8],
                      [188.3, 138.7],
                      [206.1, 101.9],
                      [171.3, 112.6]])
    L1_pixel = L1_mm / pixel_spacing

    # change units
    ann.change_units('mm')
    assert approx(L1_mm) == ann['L1']

    ann.change_units('pixel')
    assert approx(L1_pixel) == ann['L1']

    pass

def test_get_keys():

    # get input data
    ann_dict1, ann_dict2, img1, img2, pixel_spacing, units = read_test_data('MR04-019.dat')[0:6]

    # pack annotations in Annotation instances
    ann1 = Annotation(ann_dict1, pixel_spacing, units)
    keys_vert1 = ann1.get_vert_keys()
    keys_screw1 = ann1.get_keys('screw')
    keys_rod1 = ann1.get_keys('rod')

    ann2 = Annotation(ann_dict2, pixel_spacing, units)
    keys_vert2 = ann2.get_vert_keys()
    keys_screw2 = ann2.get_keys('screw')
    keys_rod2 = ann2.get_keys('rod')

    keys_common = ann1.get_common_keys(ann2)
    keys_common_vert = ann1.get_common_keys(ann2.get_vert_keys())

    # reference values
    keys_vert1_ref = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'T10', 'T11', 'T12']
    keys_screw1_ref = ['sc0', 'sc1', 'sc2', 'sc3']
    keys_rod1_ref = ['r0', 'r1']
    keys_vert2_ref = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'T8', 'T9', 'T10', 'T11', 'T12']
    keys_screw2_ref = ['sc0', 'sc1', 'sc2', 'sc3']
    keys_rod2_ref = ['r0', 'r1']
    keys_common_ref = keys_vert1_ref + keys_screw1_ref + keys_rod1_ref
    keys_common_vert_ref = keys_vert1_ref

    assert set(keys_vert1) == set(keys_vert1_ref)
    assert set(keys_rod1) == set(keys_rod1_ref)
    assert set(keys_screw1) == set(keys_screw1_ref)
    assert set(keys_vert2) == set(keys_vert2_ref)
    assert set(keys_rod2) == set(keys_rod2_ref)
    assert set(keys_screw2) == set(keys_screw2_ref)
    assert set(keys_common) == set(keys_common_ref)
    assert set(keys_common_vert) == set(keys_common_vert_ref)
    assert ann1.values(units='mm', keys=['L1']) == approx(np.array([[219.9, 128.8],
                                                                    [188.3, 138.7],
                                                                    [206.1, 101.9],
                                                                    [171.3, 112.6]]))

    assert set(ann1.get_keys('vert')) == set(ann1.get_keys('vert'))
    assert set(ann1.get_keys('screw')) == set(ann1.get_keys('screw'))
    assert set(ann1.get_keys('rod')) == set(ann1.get_keys('rod'))
    assert set(ann1.get_keys('all')) == set(ann1.get_keys('vert') + ann1.get_keys('screw') + ann1.get_keys('rod'))

    pass

# def test_values_transformed():
#
#     # FIXME: read_data() does not work. datafile is saved with spireg Annotation datatype, should be converted to mid Annotation data type
#
#     # ann1, ann2, transform = read_data('MR04-001_M12_M6.dat')
#     ann1, ann2, img1, img2, pixel_spacing, units, transform = read_test_data('Maccabi_19359.dat')
#
#     # test transform with mm units
#     transform.units = 'mm'
#     L1_transformed = ann2.values_transformed(transform, inverse=False, order='xy', units='pixel', keys=['L1'])
#     L1_transformed_inverse = ann2.values_transformed(transform, inverse=True, order='xy', units='pixel', keys=['L1'])
#
#     # reference values - use transform with pixel units
#     translation_pixel = transform.translation/ann2.pixel_spacing  # convert translation from mm to pixel
#     transform_pixel = SimilarityTransform(rotation=transform.rotation, scale=transform.scale, translation=translation_pixel)
#     L1_ref = ann2.values(order='xy', units='pixel', keys=['L1'])
#     L1_transformed_ref = transform_pixel(L1_ref)
#     L1_transformed_inverse_ref = transform_pixel.inverse(L1_ref)
#
#     assert L1_transformed == approx(L1_transformed_ref)
#     assert L1_transformed_inverse == approx(L1_transformed_inverse_ref)
#
#     # test transform with pixel units
#     transform_pixel.units = 'pixel'
#     L1_transformed = ann2.values_transformed(transform_pixel, inverse=False, order='xy', units='pixel', keys=['L1'])
#     L1_transformed_inverse = ann2.values_transformed(transform_pixel, inverse=True, order='xy', units='pixel', keys=['L1'])
#
#     assert L1_transformed == approx(L1_transformed_ref)
#     assert L1_transformed_inverse == approx(L1_transformed_inverse_ref)
#
#     pass


def test_find_uiv_liv():

    ann1, ann2, img1, img2, pixel_spacing, units, transform = read_test_data('Maccabi_19359.dat')

    # ann2.plot_annotations()  # display

    uiv, liv, vert_above_uiv, vert_below_liv = ann2.get_uiv_liv(keys_wanted=None)

    assert uiv == 'L4'
    assert liv == 'S1'
    assert set(vert_above_uiv) == set(['T11', 'T12', 'L1', 'L2', 'L3'])
    assert set(vert_below_liv) == set([])

    keys_wanted = ['L2', 'L3', 'L4', 'L5']
    uiv, liv, vert_above_uiv, vert_below_liv = ann2.get_uiv_liv(keys_wanted=keys_wanted)

    assert uiv == 'L4'
    assert liv == 'S1'
    assert set(vert_above_uiv) == set(['L2', 'L3'])
    assert set(vert_below_liv) == set([])

    keys_wanted = ['L2', 'L3']
    uiv, liv, vert_above_uiv, vert_below_liv = ann2.get_uiv_liv(keys_wanted=keys_wanted)

    assert uiv == 'L4'
    assert liv == 'S1'
    assert set(vert_above_uiv) == set(['L2', 'L3'])
    assert set(vert_below_liv) == set([])

    uiv, liv, vert_above_uiv, vert_below_liv = ann1.get_uiv_liv(keys_wanted=None)

    assert uiv is None
    assert liv is None
    assert set(vert_above_uiv) == set([])
    assert set(vert_below_liv) == set([])

    pass


if __name__ == '__main__':

    # test_create_annotation_instance()
    # test_get_values()
    # test_change_units()
    # test_get_keys()
    # test_values_transformed()
    test_find_uiv_liv()


    pass