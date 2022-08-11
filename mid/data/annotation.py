from collections.abc import MutableMapping
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from mid.data.image_processing import simple_preprocssing


class Annotation(MutableMapping):
    """Class for convenient handling of annotations.

    Parameters
    ----------
    ann : dict
        Annotation dictionary, where keys are names and values are coordinates given as [x, y].
    pixel_spacing : array like
        Pixel spacing given as (pixel_spacing_x, pixel_spacing_y).
    units : str
        Values units, one of {'pixel' , 'mm'}
    dicom_path : str, optional
        Path to Dicom file.
    metadata : dict, optional
        Annotations' metadata.
    display : bool, optional
        If True, annotation will be displayed. dicom_path must be given for display.
    """

    def __init__(self, ann, pixel_spacing, units, dicom_path=None, metadata=None, display=False, save_fig_name=None):

        assert isinstance(ann, dict), 'ann_dict must be of type dict(), got type {}'.format(type(ann))
        assert units in ['mm', 'pixel'], "units should be one of ['mm', 'pixel'], got {}".format(units)
        assert len(pixel_spacing) == 2, "len(pixel_spacing should be 2, got {}".format(len(pixel_spacing))

        self.ann = deepcopy(ann)
        self.pixel_spacing = np.array(pixel_spacing)
        self.units = units
        self.dicom_path = dicom_path
        self.metadata = metadata

        # members
        self.vert_prefix = ('C', 'T', 'L', 'S')
        self.rod_prefix = ('r',)
        self.screw_prefix = ('sc',)
        self.implant_prefix = self.screw_prefix + self.rod_prefix
        self.icl_prefix = ('icl', )
        self.femur_prefix = ('f', )
        self.vert_implant_prefix = self.vert_prefix + self.implant_prefix
        self.all_prefix = self.vert_prefix + self.rod_prefix + self.screw_prefix + self.icl_prefix + self.femur_prefix

        if display or (save_fig_name is not None):
            self.plot_annotations(display=display, save_fig_name=save_fig_name)

        return

    def __getitem__(self, key):
        return self.ann[key]

    def __setitem__(self, key, value):
        self.ann[key] = value

    def __delitem__(self, key):
        del self.ann[key]

    def __iter__(self):
        return iter(self.ann)

    def __len__(self):
        return len(self.ann)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_elements={}, ".format(len(self.ann))
        s += "pixel_spacing={}, ".format(self.pixel_spacing)
        s += "units={}".format(self.units)
        s += ")"
        return s

    def change_units(self, units):
        """Change annotations units.

        Parameters
        ----------
        units : str
            Values units, one of {'pixel' , 'mm'}
        """

        assert units in ['mm', 'pixel'], "units should be one of ['mm', 'pixel'], got {}".format(units)

        if (units != self.units):
            if units == 'mm':
                units_factor = self.pixel_spacing
            elif units == 'pixel':
                units_factor = 1. / self.pixel_spacing

            for key, value in self.items():
                value *= units_factor

            self.units = units

        return

    def values(self, order='xy', units=None, keys=None):
        """Get annotation values as an array with specified order and units.

        Parameters
        ----------
        order : str, optional
            Output array column order, one of ['xy', 'rc']
                - 'xy': [x, y], i.e. [column, row]
                - 'rc': [y, x], i.e. [row, column]
            Note that default annotation order is 'xy'.
        units : str, optinal
            Output array units, one of {'pixel' , 'mm'}.
            If None, units will not be changed.
        keys : list, optional
            List of keys of which values will be outputted.
            If None, self.keys() will be used.

        Returns
        -------
        values : ndarray
            Output annotations' array.
        """

        if keys is None:
            keys = self.keys()

        values = np.concatenate([self[key] for key in keys])  # xy, original units

        # convert units
        if (units is not None) and (units != self.units):
            if units == 'mm':
                units_factor = self.pixel_spacing
            elif units == 'pixel':
                units_factor = 1. / self.pixel_spacing

            values *= units_factor

        # convert order
        if order == 'rc':
            values = self.swap_columns(values)

        return values

    def values_transformed(self, transform, inverse=False, order='xy', units=None, keys=None):
        """Get transformed annotation values as an array with specified order and units.

        Parameters
        ----------
        transform : subclass of skimage.transform.GeometricTransform
            Transformation to be applied. Must have a 'units' attribute.
        inverse : bool, optional
            If True, an inverse transform will be applied.
        order : str, optional
            Output array column order, one of ['xy', 'rc']
                - 'xy': [x, y], i.e. [column, row]
                - 'rc': [y, x], i.e. [row, column]
            Note that default annotation order is 'xy'.
        units : str, optinal
            Output array units, one of {'pixel' , 'mm'}.
            If None, units will not be changed.
        keys : list, optional
            List of keys of which values will be outputted.
            If None, self.keys() will be used.

        Returns
        -------
        values : ndarray
            Output annotations' array.

        """

        assert hasattr(transform, 'units'), "transform must have 'units' attribute"

        # get values in xy order and transform units
        values = self.values(order='xy', units=transform.units, keys=keys)

        # apply transform
        if inverse:
            values = transform.inverse(values)
        else:
            values = transform(values)

        # convert to wanted units
        if (units is not None) and (units != transform.units):
            if units == 'mm':
                units_factor = self.pixel_spacing
            elif units == 'pixel':
                units_factor = 1. / self.pixel_spacing

            values *= units_factor

        # convert to wanted order
        if order == 'rc':
            values = self.swap_columns(values)

        return values

    def values_dict(self, order='xy', units=None, keys=None, transform=None, inverse=False, vert_anns=True):
        """Get dictionary of annotations.
        Similar to values() and values_transformed() methods, with the addition of annotation names.
        """
        if keys is None:
            keys = self.keys()

        if transform is None:
            values = self.values(order, units, keys)
        else:
            values = self.values_transformed(transform, inverse, order, units, keys)

        step = 4 if vert_anns else 2  # verts have 4 coordinates, other anns have 2
        values_dict = {key: values[(step * n):(step * n + step)] for n, key in enumerate(keys)}

        return values_dict

    def swap_columns(self, x):
        return x[:, ::-1]

    def get_keys(self, ktype):
        assert ktype in ['vert', 'rod', 'screw', 'implant', 'icl', 'femur', 'vert_implant', 'all']
        keys_allowed = eval('self.{}_prefix'.format(ktype))
        keys = [key for key in self.keys() if key.startswith(keys_allowed)]
        return keys

    def get_all_keys(self):
        keys = self.get_keys('all')
        return keys

    def get_vert_keys(self, sort=True):
        keys = self.get_keys('vert')
        if sort:
            keys = self.sort_keys_by_vert_names(keys)
        return keys


    def get_common_keys(self, ann, keys_wanted=None):
        keys_common = sorted(list(set(self).intersection(set(ann))))
        if keys_wanted is not None:
            keys_common = sorted(list(set(keys_common).intersection(set(keys_wanted))))
        return keys_common

    def sort_keys_by_vert_names(self, keys):
        """Sort keys by vertebrates names
        """
        self.vert_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                           'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                           'L1', 'L2', 'L3', 'L4', 'L5', 'L6',
                           'S1', 'S2', 'S3', 'S4', 'S5',
                           ]
        indices_ordered = list(range(len(self.vert_names)))
        zipped_sorted_ind_vert = list(zip(indices_ordered, self.vert_names))
        indices = sorted([ind for (ind, key) in zipped_sorted_ind_vert if key in keys])  # indices of input keys
        keys_ordered = [self.vert_names[ind] for ind in indices]

        return keys_ordered

    def load_dicom(self):
        if self.dicom_path is not None:
            img = sitk.ReadImage(self.dicom_path)  # sitk.Image
            img = sitk.GetArrayFromImage(img).squeeze()  # ndarray
        else:
            img = None
        return img

    def plot_annotations(self, fontsize=8, plot_lines=False, display=True, save_fig_name=None):
        """Plot annotations on top of image.

        Parameters
        ----------
        fontsize : int, optional
            Labels font size.
        plot_lines : bool, optional
            If True, lines will be plotted between pairs of annotations.

        Returns
        -------
        None
        """

        # preprocess
        # if self.dicom_path is None:
        #     raise Value('dicom_path must be valid in order to plot')

        change_units = False
        if self.units != 'pixel':
            change_units = True
            units_orig = self.units
            self.change_units('pixel')  # for plotting units must be in pixels

        # unpack data
        img_path = self.dicom_path
        ann_dict = self.ann

        title_str = dict2string(self.metadata)
        fig = plot_annotations(img_path, ann_dict, fontsize, plot_lines, title_str=title_str, show=display, save_fig_name=save_fig_name)

        if change_units:
            self.change_units(units_orig)

        return fig

def dict2string(d, sep=' | ', n_new_line=2):

    s = ''

    if d is not None:

        counter = 0
        new_line = True
        end_line = False
        for key, val in d.items():

            if val is None:
                continue

            counter += 1

            if end_line:
                s += '\n'
                end_line = False
                new_line = True

            if new_line:  # TODO: maybe can unify new/end line logic
                s += '{}: {}'.format(key, val)
                new_line = False

            else:  # add seperator
                s += '{}'.format(sep)
                s += '{}: {}'.format(key, val)
                if counter % n_new_line == 0:
                    end_line = True

    return s

def plot_annotations(img_path, ann_dict, fontsize=8, plot_lines=False, fig=None, preprocess='clahe1',
                     colors=['r', 'b', 'm', 'c'], marker='x', title_str='', show=True, save_fig_name=None):
    """
    Plot annotations on top of image.

    Parameters
    ----------
    img_path : str
        Image file path
    ann_dict : dict
        Annotation dictionary
    fontsize : int, optional
        Labels font size.
    plot_lines : bool, optional
        If True, lines will be plotted between pairs of annotations.
    colors : str or Iterable of str, optional
        Wanted colors.
    marker : str, optional
        Wanted marker

    Returns
    -------
    None
    """

    # show image
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

        if img_path is not None:
            # read image as ndarray
            img = sitk.ReadImage(img_path)  # sitk.Image
            nda = sitk.GetArrayFromImage(img).squeeze()  # ndarray
            if preprocess is not None:
                nda = simple_preprocssing(nda, preprocess, keep_input_dtype=False, display=False)
            ax.imshow(nda, cmap='gray')
        else:
            ax.invert_yaxis()

        ax.axis('scaled')

    # plot anns
    counter = 0
    for ann_name, ann in ann_dict.items():
        color = colors[counter % len(colors)]  # alternate between colors
        fig = plot_ann(fig, ann, ann_name, color, marker, fontsize, plot_lines)
        counter += 1

    if show or save_fig_name:
        plt.title(title_str, fontsize=16)
        fig.get_axes()[0].axis('scaled')
        plt.tight_layout()

    if show:
        fig.show()

    if save_fig_name:
        fig.savefig(save_fig_name)

    return fig

def plot_ann(fig, ann, text, color='r', marker='x', fontsize=8, plot_lines=False):

    ax = fig.get_axes()[0]
    ax.plot(ann[:, 0], ann[:, 1], '{}{}'.format(color, marker), mfc='none')
    ax.text(ann[:, 0].mean(), ann[:, 1].mean(), text,
            verticalalignment='center', horizontalalignment='center',
            # transform=ax.transAxes,
            color=color, fontsize=fontsize)

    # plot lines between pairs of annotations
    if plot_lines:
        for n in range(0, ann.shape[0], 2):  # jump 2 rows at a time
            ax.plot(ann[n:n + 2, 0], ann[n:n + 2, 1], '{}--'.format(color))

    return fig