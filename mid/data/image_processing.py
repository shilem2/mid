import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask
from skimage import exposure, img_as_float


def adjust_dynamic_range(img, vmin=0, vmax=255, dtype=np.float32):

    img_out = img.astype(np.float32)  # for middle calculations
    img_out -= vmin
    img_out *= vmax / img_out.max()
    img_out = img_out.astype(dtype)

    return img_out


def generate_simple_mask(img, anns, margin=20, display=False):

    # get extreme anns
    left = int(anns[:, 0].min())
    top = int(anns[:, 1].min())
    right = int(anns[:, 0].max())
    bottom = int(anns[:, 1].max())

    # get image size
    height, width = img.shape[:2]

    # add margins, keep values inside image boudaries
    left = np.maximum(0, left - margin)
    top = np.maximum(0, top - margin)
    right = np.minimum(width, right + margin)
    bottom = np.minimum(height, bottom + margin)

    # generate mask
    mask = np.zeros_like(img)
    mask[top:bottom, left:right] = 1

    # apply mask
    img_masked = mask * img

    if display:
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title('image', fontsize=22)
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('mask', fontsize=22)
        plt.subplot(133)
        plt.imshow(img_masked, cmap='gray')
        plt.title('masked image', fontsize=22)
        plt.show()

    return img_masked

def simple_preprocssing(img, process_type, keep_input_dtype=True, display=False):
    """
    Simple pre processing of image gray levels
    Parameters
    ----------
    img : ndarray
        Input image.
    process_type : str
        One of {'unsharp_mask', 'clahe1', 'clahe2'}
    keep_input_dtype : bool, optional
        If True, img_out will have same dtype as img.
    display : bool, optional
        If True, image before and after processing will be displayed.

    Returns
    -------
    img_out : ndarray
        Processed image.
    """

    if process_type not in {'unsharp_mask', 'clahe1', 'clahe2', 'none'}:
        raise ValueError("process must be one of ['unsharp_mask', 'clahe1', 'clahe2', 'none'], got {}".format(process_type))

    if process_type == 'unsharp_mask':
        img_out = unsharp_mask(img, radius=20, amount=1, preserve_range=True)
    elif process_type == 'clahe1':
        img_out = exposure.equalize_adapthist(img, kernel_size=None, clip_limit=0.03, nbins=256)
    elif process_type == 'clahe2':
        img_out = exposure.equalize_adapthist(img, kernel_size=140, clip_limit=0.03, nbins=256)

    if keep_input_dtype:
        img_out = adjust_dynamic_range(img, vmin=img.min(), vmax=img.max(), dtype=img.dtype)

    if display:
        fig, axes = plt.subplots(nrows=1, ncols=2,
                                 sharex=True, sharey=True, figsize=(10, 10))
        ax = axes.ravel()
        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].set_title('Original image', fontsize=22)
        ax[1].imshow(img_out, cmap=plt.cm.gray)
        ax[1].set_title('Enhanced image, {}'.format(process_type), fontsize=22)
        for a in ax:
            a.axis('off')
        fig.tight_layout()
        plt.show()

    return img_out
