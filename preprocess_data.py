"""
@brief This script is used for the pre-processing of the data
       to be used for training and validation of 3d fetal brain segmentation.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import os
import csv
import numpy as np
import nibabel as nib
import random
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import binary_fill_holes
from definitions import *


# Masking
NUM_ITER_MASK_DILATION = 5
NO_SKULL_STRIP = False  # for having non skull-stripped images for figures
# SKULL_STRIP_AUG = True  # every image is save twice: w and \w skull stripping

SAVE_PATH = os.path.join(DATA_FOLDER, 'fetal_training_Mar22')
LABELS_SUPERSET_MAP = {
    9: [LABELS['wm'], LABELS['corpus_callosum']],
}


class Sample:
    def __init__(self, patient_id, session_id, srr_path, mask_path,
                 parcellation_path=None):
        self.patient_id = patient_id
        self.session_id = session_id
        self.srr_path = srr_path
        check_path(srr_path)
        self.mask_path = mask_path
        check_path(mask_path)
        self.parcellation_path = parcellation_path
        if parcellation_path is not None:
            check_path(parcellation_path)

    @classmethod
    def from_folder(cls, folder_path, use_parcellation=True):
        """
        Auto creation of Sample when:
        0. the study folders have the name convention 'PatientName_StudyID(_whatever)'
        1. SRR is called 'srr.nii.gz'
        2. mask is called 'mask.nii.gz'
        3. parcellation is called 'parcellation.nii.gz'
        :param folder_path: path to the study
        :return: a sample
        """
        folder_name = os.path.split(folder_path)[1]
        split_folder_name = folder_name.split('_')
        pat_id = split_folder_name[0]
        if len(split_folder_name) > 1:
            study_id = folder_name.split('_')[1]
        else:
            study_id = 'Study1'
        srr_path = os.path.join(folder_path, 'srr.nii.gz')
        if not os.path.exists(srr_path):
            srr_path = os.path.join(folder_path, 'srr_template.nii.gz')
        mask_path = os.path.join(folder_path, 'mask.nii.gz')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(folder_path, 'srr_template_mask.nii.gz')
        if use_parcellation:
            parc_path = os.path.join(folder_path, 'parcellation.nii.gz')
        else:
            parc_path = None
        study = cls(pat_id, study_id, srr_path, mask_path, parc_path)
        return study

    @property
    def id(self):
        return '%s_%s' % (self.patient_id, self.session_id)

    @property
    def save_folder(self):
        save_study_path = os.path.join(SAVE_DATA_FOLDER, self.id)
        return save_study_path


def get_normalized_studies(group_folder, use_parcellation=True):
    """
    Auto creation of Sample when:
    0. the study folders have the name convention 'PatientName_StudyID'
    1. SRR is called 'srr.nii.gz'
    2. mask is called 'mask.nii.gz'
    3. parcellation is called 'parcellation.nii.gz' (if used)
    :param group_folder:
    :return: a list of samples
    """
    samples = []
    sub_folders = [f for f in os.listdir(group_folder) if not '.' in f]
    for study_folder in sub_folders:
        folder_path = os.path.join(group_folder, study_folder)
        study = Sample.from_folder(folder_path, use_parcellation=use_parcellation)
        samples.append(study)
    print('Found %d studies in %s' % (len(samples), group_folder))
    return samples


def get_Thomas_group_studies():
    samples_list = []
    count = 0
    folder_path = DATA_FOLDER_THOMAS_GROUP1
    study_folders = [
        f for f in os.listdir(folder_path)
        if not '.' in f
    ]
    for study_folder in study_folders:
        if 'Study' in study_folder:
            study_nb = 'Study%d' % int(study_folder.split('Study')[1])
            pat_id = study_folder.split('Study')[0].replace('_', '')
        else:
            study_nb ='Study1'
            pat_id = study_folder.replace('_', '')
        study_path = os.path.join(folder_path, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d abnormal brain studies in %s." % (count, folder_path))
    return samples_list


def get_Thomas_group_Mar20_studies():
    """
    14 abnormal cases that were used for testing in the US meets MRI workshop in 2020.
    It includes 2 Spina Bifida cases, and 12 cases with other CNS abnormalities.
    All MRI were done at UZL.
    """
    samples_list = []
    count = 0
    folder_path = DATA_FOLDER_THOMAS_GROUP2
    study_folders = [
        f for f in os.listdir(folder_path)
        if not '.' in f
    ]
    for study_folder in study_folders:
        pat_id = study_folder.replace('_', '')
        study_nb ='Study1'
        study_path = os.path.join(folder_path, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d abnormal brain studies in %s." % (count, folder_path))
    return samples_list


def get_CDH_Doaa_studies():
    samples_list = []
    count = 0
    subgroups_folders = [
        'CDH_Doaa_Dec19',
    ]
    for subgroup in subgroups_folders:
        folder_path = os.path.join(DATA_FOLDER_MICHAEL_GROUP, subgroup)
        study_folders = [
            f for f in os.listdir(folder_path)
            if not '.' in f
        ]
        for study_folder in study_folders:
            pat_id = study_folder.replace('-', '_').split('_')[0]
            study_nb = 'Study1'
            study_path = os.path.join(folder_path, study_folder)
            srr_path = os.path.join(study_path, 'srr.nii.gz')
            mask_path = os.path.join(study_path, 'mask.nii.gz')
            parc_path = os.path.join(study_path, 'parcellation.nii.gz')
            new_sample = Sample(
                pat_id, study_nb, srr_path, mask_path, parc_path)
            samples_list.append(new_sample)
            count += 1
        print("Found %d CDH studies in %s." % (count, folder_path))
    return samples_list


def get_CDH_Doaa_longitudinal():
    samples_list = []
    count = 0
    study_folders = [
        f for f in os.listdir(DATA_FOLDER_CDH_LONG)
        if not '.' in f
    ]
    for study_folder in study_folders:
        pat_id = study_folder.replace('_', '')
        study_nb = 'Study1'
        study_path = os.path.join(DATA_FOLDER_CDH_LONG, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d CDH studies in %s." % (count, DATA_FOLDER_CDH_LONG))
    return samples_list


def get_CDH_Doaa_longitudinal2():
    samples_list = []
    count = 0
    study_folders = [
        f for f in os.listdir(DATA_FOLDER_CDH_LONG2)
        if not '.' in f
    ]
    for study_folder in study_folders:
        pat_id = study_folder.replace('_', '')
        study_nb = 'Study1'
        study_path = os.path.join(DATA_FOLDER_CDH_LONG2, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d CDH studies in %s." % (count, DATA_FOLDER_CDH_LONG2))
    return samples_list


def get_Controls_Doaa():  # 7 new controls
    samples_list = []
    count = 0
    folder_path = os.path.join(DATA_FOLDER_MICHAEL_GROUP, 'Controls_Doaa_Oct20_MA')
    study_folders = [
        f for f in os.listdir(folder_path)
        if not '.' in f
    ]
    for study_folder in study_folders:
        pat_id = study_folder.replace('_MA', '').replace('_CC', '').replace('_', '')
        study_nb = 'Study1'
        study_path = os.path.join(folder_path, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d control studies in %s." % (count, folder_path))
    return samples_list


def get_SB_Fred():
    samples_list = []
    count = 0
    folder_path = SB_FRED
    study_folders = [
        f for f in os.listdir(folder_path)
        if not '.' in f
    ]
    for study_folder in study_folders:
        pat_id = study_folder.replace('_', '')
        study_nb = 'Study1'
        study_path = os.path.join(folder_path, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d control studies in %s." % (count, folder_path))
    return samples_list


def get_SB_Fred2():
    samples_list = []
    count = 0
    folder_path = SB_FRED2
    study_folders = [
        f for f in os.listdir(folder_path)
        if not '.' in f
    ]
    for study_folder in study_folders:
        pat_id = study_folder.replace('_', '')
        study_nb = 'Study1'
        study_path = os.path.join(folder_path, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d control studies in %s." % (count, folder_path))
    return samples_list


def get_Harvard_studies():
    samples_list = []
    count = 0
    folder_path = DATA_FOLDER_HARVARD_GROUP
    # Get all the folders.
    # Each folder corresponds to one patient.
    study_folders = [l for l in os.listdir(folder_path)
                  if l.startswith('Harvard')]
    for study_folder in study_folders:
        pat_id = study_folder.split('_')[0]
        study_nb = 'Study1'
        study_path = os.path.join(folder_path, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d studies in %s." % (count, folder_path))
    return samples_list


def get_SB_atlas_studies():
    samples_list = []
    count = 0
    folder_path = DATA_FOLDER_SPINA_BIFIDA_ATLAS
    # Get all the folders.
    # Each folder corresponds to one patient.
    study_folders = [l for l in os.listdir(folder_path)
                  if l.startswith('fetal')]
    for study_folder in study_folders:
        pat_id = study_folder.replace('_', '')
        study_nb = 'Study1'
        study_path = os.path.join(folder_path, study_folder)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        parc_path = os.path.join(study_path, 'parcellation.nii.gz')
        new_sample = Sample(
            pat_id, study_nb, srr_path, mask_path, parc_path)
        samples_list.append(new_sample)
        count += 1
    print("Found %d studies in %s." % (count, folder_path))
    return samples_list


def get_dhcp_studies(use_parcellation=True):
    # Add the 112 studies from the developing Human Connectom Project (data release 2)
    # with maximum gestational ages of 38 weeks.
    dhcp_folder = DATA_FOLDER_DHCP_GROUP
    samples_list = []
    # Get all the folders in the dHCP project.
    # Each folder corresponds to one patient and can contain several studies.
    folders_sub = [l for l in os.listdir(dhcp_folder)
                  if l.startswith('sub')]
    for sub_f in folders_sub:
        patient_id = sub_f.split('_')[0]
        session_id = sub_f.split('_')[1]
        study_path = os.path.join(dhcp_folder,
                                sub_f)
        srr_path = os.path.join(study_path, 'srr.nii.gz')
        mask_path = os.path.join(study_path, 'mask.nii.gz')
        # DO NOT USE THE SEGMENTATIONS
        if use_parcellation:
            parc_path = os.path.join(study_path, 'parcellation_corrected.nii.gz')
        else:
            parc_path = None
        new_sample = Sample(
            patient_id=patient_id,
            session_id=session_id,
            srr_path=srr_path,
            mask_path=mask_path,
            parcellation_path=parc_path,
        )
        samples_list.append(new_sample)
    print("Found %d studies in the preterms dHCP dataset." %
          len(samples_list))
    return samples_list


def preprocessing_pipeline(sample):
    def mask_from_seg(seg_np, initial_mask_np):
        class_present = np.unique(seg_np).tolist()
        assert LABELS['external_csf'] in class_present, \
            "mask_from_seg works only if the extra-axial CSF is present in the manual segmentation."

        # First make sure the ext-csf is closed
        new_mask = (seg_np > 0)
        eroded_mask = binary_erosion(initial_mask_np, iterations=3)
        new_mask[eroded_mask] = True

        # Second compute the new mask
        new_mask = binary_fill_holes(new_mask).astype(np.uint8)

        return new_mask

    assert isinstance(sample, Sample)

    print('\n\033[93mStart preprocessing of %s\033[0m' % sample.save_folder)

    # Load the SRR image, mask and segmentation
    srr_nii = load_data(sample.srr_path)
    srr_np = srr_nii.get_fdata().astype(np.float32)
    if np.count_nonzero(np.isnan(srr_np)) > 0:
        srr_np[np.isnan(srr_np)] = 0

    mask_nii = load_data(sample.mask_path)
    mask_np = mask_nii.get_fdata().astype(np.uint8)
    if np.count_nonzero(np.isnan(mask_np)) > 0:
        mask_np[np.isnan(mask_np)] = 0

    parcellation_nii = load_data(sample.parcellation_path)
    parcellation_np = parcellation_nii.get_fdata().astype(np.uint8)
    if np.count_nonzero(np.isnan(parcellation_np)) > 0:
        parcellation_np[np.isnan(parcellation_np)] = 0

    # Dilation of the mask and keep only the largest component
    mask_np[parcellation_np > 0] = 1
    mask_dilated_np = binary_dilation(mask_np, iterations=NUM_ITER_MASK_DILATION)
    mask_dilated_np = keep_only_largest_component_of_mask(mask_dilated_np)
    parcellation_np[mask_dilated_np == 0] = 0  # remove potential false positives
    mask_np[mask_dilated_np == 0] = 0

    # If extra-axial CSF is present,
    # compute a new mask based on the segmentation
    class_present = np.unique(parcellation_np).tolist()
    if LABELS['external_csf'] in class_present:
        print('Compute the brain mask using the parcellation that contains extra-axial CSF.')
        mask_np = mask_from_seg(parcellation_np, initial_mask_np=mask_np)

    # pre-process the SRR
    # clip percentile 99.9%
    p_999 = np.percentile(srr_np, 99.9)
    srr_np[srr_np > p_999] = p_999

    if not NO_SKULL_STRIP:
        # Skull strip
        srr_np =  skull_strip(srr_np, mask_dilated_np, dilation_iter=0)

    # Pad to a large size
    # (this is to make sure the brain will be well centered during cropping)
    min_size_pad = [MIN_SIZE[i] + 50 for i in range(3)]
    srr_np = pad_if_needed(srr_np, min_size=min_size_pad)
    parcellation_np = pad_if_needed(parcellation_np, min_size=min_size_pad)
    mask_np = pad_if_needed(mask_np, min_size=min_size_pad)

    # Crop around the mask to make sure the brain is well centered
    crop_srr = crop_around_mask(srr_np, mask_np, patch_size=MIN_SIZE)
    crop_parc = crop_around_mask(parcellation_np, mask_np, patch_size=MIN_SIZE)
    crop_mask = crop_around_mask(mask_np, mask_np, patch_size=MIN_SIZE)

    # Partial segmentation preprocessing
    # Compute the set of missing labels
    missing_labels = []
    for label in list(LABELS.values()):
        num = np.sum(crop_parc == label)
        if num == 0:
            missing_labels.append(label)
    # Skip if no missing labels
    if len(missing_labels) > 0:
        # Set the superset label
        if missing_labels == [LABELS['corpus_callosum']]:
            if 'sub' in sample.patient_id:
                superset_label = 9  # wm + cc
                print('Convert WM into WM+CC')
                crop_parc[crop_parc == LABELS['wm']] = superset_label
        else:
            import collections
            superset_label = None
            for label in list(LABELS_SUPERSET_MAP.keys()):
                # Find the appropriate superset label
                if collections.Counter(LABELS_SUPERSET_MAP[label]) == collections.Counter(missing_labels):
                    superset_label = label
            assert superset_label is not None, 'Superset label not found for %s' % str(missing_labels)
            crop_parc[np.logical_and(crop_parc == 0, crop_mask > 0)] = superset_label

    srr_nii = nib.Nifti1Image(crop_srr, srr_nii.affine, srr_nii.header)
    parcellation_nii = nib.Nifti1Image(crop_parc, mask_nii.affine, mask_nii.header)
    mask_nii = nib.Nifti1Image(crop_mask, mask_nii.affine, mask_nii.header)

    # Save everything
    study_path = sample.save_folder
    if not os.path.exists(study_path):
        os.mkdir(study_path)
    srr_path = os.path.join(study_path, 'srr.nii.gz')
    nib.save(srr_nii, srr_path)
    parcellation_path = os.path.join(study_path, 'parcellation.nii.gz')
    nib.save(parcellation_nii, parcellation_path)
    mask_path = os.path.join(study_path, 'mask.nii.gz')
    nib.save(mask_nii, mask_path)

    print("%s has been pre-processed" % study_path)


if __name__ == '__main__':
    random.seed(12)

    # Get info for all the studies required for the selected config
    samples = []

    print('Prepare March 2022 dataset')
    # Data Model Dec 2021
    # Zurich data
    samples += get_normalized_studies(CORRECTED_ZURICH_DATA_DIR)  # 30
    samples += get_normalized_studies(EXCLUDED_ZURICH_DATA_DIR)  # 8
    samples += get_normalized_studies(FETA_CHALLENGE_DIR)  # 40
    # Atlas data
    samples += get_Harvard_studies()  # 18
    samples += get_SB_atlas_studies()  # 15
    samples += get_normalized_studies(DATA_FOLDER_CHINESE_ATLAS)  # 14
    # Leuven data
    samples += get_normalized_studies(CDH_LEUVEN_TESTINGSET)  # 19
    samples += get_normalized_studies(DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG)  # 7
    samples += get_Thomas_group_studies()  # 23
    samples += get_SB_Fred()  # 46
    # From here: New in Mar 2022
    samples += get_normalized_studies(DATA_FOLDER_VIENNA_MMC)  # 11
    samples += get_normalized_studies(DATA_FOLDER_KCL_CONTROLS)  # 29
    samples += get_normalized_studies(DATA_FOLDER_UCLH_MMC_2)  # 47
    samples += get_normalized_studies(TESTING_ZURICH_DATA_DIR)  # 10
    # Leuven data
    samples += get_normalized_studies(DATA_FOLDER_LEUVEN_MMC)  # 28
    samples += get_SB_Fred2()  # 15
    samples += get_Thomas_group_Mar20_studies()  # 14
    samples += get_CDH_Doaa_longitudinal()  # 19
    samples += get_CDH_Doaa_longitudinal2()  # 50
    samples += get_normalized_studies(DATA_FOLDER_CDH)  # 16
    samples += get_normalized_studies(DATA_FOLDER_CONTROLS_WITH_EXTCSF_MA)  # 26

    SAVE_DATA_FOLDER = SAVE_PATH

    if not os.path.exists(SAVE_DATA_FOLDER):
        os.mkdir(SAVE_DATA_FOLDER)
    print('\nTotal: %d studies to preprocess' % len(samples))

    # Pre-processing for our studies
    for s in samples:
        if os.path.exists(s.save_folder):
            print('\n', s.save_folder, 'already exists. Skip preprocessing.')
            continue
        preprocessing_pipeline(s)
