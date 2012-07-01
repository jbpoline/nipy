#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script requires the nipy-data package to run. It is an example of
using a general linear model in single-subject fMRI data analysis
context. Two sessions of the same subject are taken from the FIAC'05
dataset.

Usage:
  python compute_fmri_contrast [contrast_vector]

Example:
  python compute_fmri_contrast 1, -1, 1, -1

  An activation image is displayed.

Author: Alexis Roche, Bertrand Thirion, 2009--2012.
"""
import numpy as np
import sys
from nibabel import load as load_image
import pylab as plt

from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.glm import GeneralLinearModel, data_scaling
from nipy.utils import example_data



# Optional argument - default value below
cvect = [1, 0, 0, 0]
if len(sys.argv) > 1:
    print sys.argv[1]
    try:
        tmp = sys.argv[1].split(',')
        cvect = [float(argval) for argval in tmp]
        if len(cvect) != 4:
            quit()
    except: # any error ? :
        print('usage : python %s 1x4-contrast') % sys.argv[0] 
        print('where 1x4-contrast is something like 1,0,0,0')
        quit()


# Input files
fmri_files = [example_data.get_filename('fiac', 'fiac0', run)
              for run in ['run1.nii.gz', 'run2.nii.gz']]
design_files = [example_data.get_filename('fiac', 'fiac0', run)
                for run in ['run1_design.npz', 'run2_design.npz']]
mask_file = example_data.get_filename('fiac', 'fiac0', 'mask.nii.gz')

# Get design matrix as numpy array
print('Loading design matrices...')
X = [np.load(f)['X'] for f in design_files]

# Get multi-session fMRI data
print('Loading fmri data...')
Y = [load_image(f) for f in fmri_files]

# Get mask image
print('Loading mask...')
mask = load_image(mask_file)
mask_array, affine = mask.get_data() > 0, mask.get_affine()

# GLM fitting
print('Starting fit...')
glms = []
for x, y in zip(X, Y):
    glm = GeneralLinearModel(x)
    data, mean = data_scaling(y.get_data()[mask_array].T)
    glm.fit(data, 'ar1')
    glms.append(glm)

# Compute the required contrast
print('Computing test contrast image...')
nregressors = X[0].shape[1]
## should check that all design matrices have the same
c = np.zeros(nregressors)
c[0:4] = cvect
z_vals = (glms[0].contrast(c) + glms[1].contrast(c)).z_score()

# Show Zmap image
z_map = mask_array.astype(np.float)
z_map[mask_array] = z_vals
mean_map = mask_array.astype(np.float)
mean_map[mask_array] = mean
plot_map(z_map,
         affine,
         anat=mean_map,
         anat_affine=affine,
         cmap=cm.cold_hot,
         threshold=2.5,
         black_bg=True)
plt.show()
