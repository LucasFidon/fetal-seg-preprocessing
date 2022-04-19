import os
import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# HOME_FOLDER = '/'
HOME_FOLDER = '/home/lucasf/'
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'workspace')
DATA_FOLDER = os.path.join(HOME_FOLDER, 'data')
BASE_FOLDER = os.path.join(DATA_FOLDER, 'Fetal_SRR_and_Seg')

# ATLAS FOLDERS
DATA_FOLDER_HARVARD_GROUP = os.path.join(
    DATA_FOLDER,
    'fetal_brain_atlases',
    'Gholipour2017_atlas_NiftyMIC_preprocessed_corrected',
)
DATA_FOLDER_SPINA_BIFIDA_ATLAS = os.path.join(
    DATA_FOLDER,
    'spina_bifida_atlas',
)
DATA_FOLDER_CHINESE_ATLAS = os.path.join(
    DATA_FOLDER,
    'fetal_brain_atlases',
    'FBA_Chinese_main_preprocessed_corrected',
)

# OTHER DATASETS
DATA_FOLDER_NADA_GROUP = os.path.join(BASE_FOLDER, 'SRR_and_Seg_Nada_cases_group')
DATA_FOLDER_MICHAEL_GROUP = os.path.join(BASE_FOLDER, 'SRR_and_Seg_Michael_cases_group')
DATA_FOLDER_CDH_LONG = os.path.join(BASE_FOLDER, 'Doaa_brain_longitudinal_SRR_and_Seg_MA')
DATA_FOLDER_CDH_LONG2 = os.path.join(BASE_FOLDER, 'Doaa_brain_longitudinal_SRR_and_Seg_2')
DATA_FOLDER_DHCP_GROUP = os.path.join(BASE_FOLDER, 'dHCP_neonats_upto38GA_correctedLF')
DATA_FOLDER_CONTROLS_WITH_EXTCSF_MA = os.path.join(DATA_FOLDER_NADA_GROUP, 'Controls_with_extcsf_MA')
DATA_FOLDER_NADA_CONTROLS2_MA_CC = os.path.join(DATA_FOLDER_NADA_GROUP, 'Nada_controls2_MA_CC')
DATA_FOLDER_NADA_CONTROLS2_DOAA = os.path.join(DATA_FOLDER_NADA_GROUP, 'Nada_controls2_Doaa')
DATA_FOLDER_CDH = os.path.join(DATA_FOLDER_NADA_GROUP, 'CDH')
DATA_FOLDER_LEUVEN_MMC = os.path.join(DATA_FOLDER_NADA_GROUP, 'Leuven_MMC')
DATA_FOLDER_UCLH_MMC = os.path.join(DATA_FOLDER_NADA_GROUP, 'UCLH_MMC')
DATA_FOLDER_UCLH_MMC_2 = os.path.join(DATA_FOLDER_NADA_GROUP, 'UCLH_MMC_2')
DATA_FOLDER_VIENNA_MMC = os.path.join(DATA_FOLDER_NADA_GROUP, 'vienna_MMC_unoperated')
DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG = os.path.join(DATA_FOLDER_NADA_GROUP, 'Controls_2_partial')
DATA_FOLDER_THOMAS_GROUP1 = os.path.join(DATA_FOLDER_MICHAEL_GROUP, 'Abnormal_cases')
DATA_FOLDER_THOMAS_GROUP2 = os.path.join(DATA_FOLDER_MICHAEL_GROUP, 'Abnormal_cases_Mar20')

# TESTING DATASETS
CDH_LEUVEN_TESTINGSET = os.path.join(  # 19 CDH cases
    DATA_FOLDER_MICHAEL_GROUP,
    'CDH_Doaa_Aug20',
)
SB_FRED = os.path.join(  # 46 SB cases
    BASE_FOLDER,
    'SRR_and_Seg_Frederic_cases_group',
    'SB_Fred_corrected_partial',
)
SB_FRED2 = os.path.join(  # 15 SB cases
    BASE_FOLDER,
    'Fred_additional_cases_Sept2021',
)

# FETA DATA (not skull stripped but with auto brain mask)
CORRECTED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSet')
EXCLUDED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSetExcluded')
TESTING_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TestingSet')
FETA_CHALLENGE_DIR = os.path.join(DATA_FOLDER, 'FetalDataFeTAChallengeIRTK_Jun21_corrected')

# KCL DATA
DATA_FOLDER_KCL_CONTROLS = os.path.join(BASE_FOLDER, 'SRR_and_Seg_KCL', 'Control')
DATA_FOLDER_KCL_VM = os.path.join(BASE_FOLDER, 'SRR_and_Seg_KCL', 'Ventriculomegaly')


LABELS = {
    'wm': 1,
    'csf': 2,  # inner CSF space
    'cerebellum': 3,
    'external_csf': 4,
    'cortical_gm': 5,
    'deep_gm': 6,
    'brainstem': 7,
    'corpus_callosum': 8,
    # 'hippocampus': 9,
    'background': 0,
}
FETA_CHALLENGE_LABELS = {
    'wm': 3,
    'csf': 4,  # ventricles
    'cerebellum': 5,
    'external_csf': 1,
    'cortical_gm': 2,
    'deep_gm': 6,
    'brainstem': 7,
    'background': 0,
}
GA = [ga for ga in range(21, 39)]
MIN_SIZE = [128, 160, 128]
CROP_MARGIN_MIN = 2


def run_command(cmd):
    """
    Run a command without printing the output in the terminal.
    :param cmd: str; command to run.
    """
    # Run the command without printing the logs on the screen
    os.system('%s > /dev/null 2>&1' % cmd)
    # os.system(cmd)

def check_path(path):
    assert os.path.exists(path), "cannot find %s" % path

def load_data(path):
    data = nib.load(path)
    return data

def skull_strip(img, mask, dilation_iter=3):
    if dilation_iter > 0:
        dilated_mask = binary_dilation(
        mask, iterations=dilation_iter)
        img[dilated_mask == 0] = 0.
    else:
        img[mask == 0] = 0.
    return img


def crop_around_mask(img, mask, return_coordinates=False, patch_size=MIN_SIZE):
    """
    Crop img around mask with a margin.
    All axis dimension of the cropped image will not be smaller
    than the one of the patch size.
    :param img: numpy array
    :param mask: numpy array
    :param return_coordinates: bool
    :param patch_size:
    :return:
    """
    def try_extend_if_need(axis_min, axis_max, axis_patch_size, axis_dim):
        if axis_max - axis_min >= axis_patch_size:
            return axis_min, axis_max
        else:
            extra = axis_patch_size - (axis_max - axis_min)  # what needs to be added
            axis_min -= extra // 2
            axis_min = max(axis_min, 0)
            axis_max = axis_min + axis_patch_size
            axis_max = min(axis_max, axis_dim)
            # If it still does not fit it means axis_max == axis_dim already
            # and the initial shift of axis_min was not enough.
            # Rq: it is better to pad the image before to avoid being in this case.
            if axis_max - axis_min < axis_patch_size:
                print('Warning: the brain might not be well centered after cropping. '
                      'Pad your image please.')
                axis_min = axis_max - axis_patch_size
        return axis_min, axis_max
    assert img.shape == mask.shape, "image and mask do not have the same shape"
    assert np.all(np.array(img.shape) >= patch_size), \
        " Image of shape %s is too small to fit a patch of shape %s. Try to pad the image." % \
        (str(img.shape), str(patch_size))
    # Get the number of foreground pixels
    num_fg = np.sum(mask)
    x_dim, y_dim, z_dim = tuple(img.shape)
    # Crop around the mask if present
    assert num_fg > 0, "The segmentation contains only background."
    x_fg, y_fg, z_fg = np.where(mask >= 1)
    # Get the extremal acceptable coordinates for cropping
    x_min = max(int(np.min(x_fg)) - CROP_MARGIN_MIN, 0)
    x_max = min(int(np.max(x_fg)) + CROP_MARGIN_MIN, x_dim)
    x_min, x_max = try_extend_if_need(x_min, x_max, patch_size[0], x_dim)
    y_min = max(int(np.min(y_fg)) - CROP_MARGIN_MIN, 0)
    y_max = min(int(np.max(y_fg)) + CROP_MARGIN_MIN, y_dim)
    y_min, y_max = try_extend_if_need(y_min, y_max, patch_size[1], y_dim)
    z_min = max(int(np.min(z_fg)) - CROP_MARGIN_MIN, 0)
    z_max = min(int(np.max(z_fg)) + CROP_MARGIN_MIN, z_dim)
    z_min, z_max = try_extend_if_need(z_min, z_max, patch_size[2], z_dim)
    # Crop the image
    crop_img = img[x_min:x_max, y_min:y_max, z_min:z_max]
    if return_coordinates:
        coords = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        return crop_img, coords
    else:
        return crop_img


def pad_if_needed(array_3d, min_size=MIN_SIZE, return_padding_values=False):
    """
    Pad an array with zeros if needed so that each of its dimensions
    is at least equal to MIN_SIZE.
    :param array_3d: numpy array to be padded if needed
    :return: padded array.
    """
    shape = array_3d.shape
    need_padding = np.any(shape < np.array(min_size))
    if not need_padding:
        pad_list = [(0, 0)] * 3
        if return_padding_values:
            return array_3d, np.array(pad_list)
        else:
            return array_3d
    else:
        pad_list =[]
        for dim in range(3):
            diff = min_size[dim] - shape[dim]
            if diff > 0:
                margin = diff // 2
                pad_dim = (margin, diff - margin)
                pad_list.append(pad_dim)
            else:
                pad_list.append((0, 0))
        padded_array = np.pad(
            array_3d,
            pad_list,
            'constant',
            constant_values = [(0,0), (0, 0), (0, 0)],
        )
        if return_padding_values:
            return padded_array, np.array(pad_list)
        else:
            return padded_array


def keep_only_largest_component_of_mask(mask):
    structure = np.ones((3, 3, 3), dtype=np.int)
    labeled, ncomp = label(mask, structure)
    size_comp = [
        np.sum(labeled == l) for l in range(1, ncomp + 1)
    ]
    first_largest_comp = np.argmax(size_comp)
    label_first = first_largest_comp + 1
    # Set all the components that are not the largest to 0
    # and the largest component to 1.
    for i in range(1, ncomp + 1):
        if i == label_first:
            labeled[labeled == i] = 1
        else:
            labeled[labeled == i] = 0
    return labeled
