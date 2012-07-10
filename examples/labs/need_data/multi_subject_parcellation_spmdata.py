# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script contains an example of a multi'subject parcellation
on a series of 3D contrasts taken from the imagen database.
See multi_subject_parcellation for an example with simulated data.
"""
print __doc__

import os
from tempfile import mkdtemp
import numpy as np
import nibabel as nib
import nipy.labs.spatial_models.hierarchical_parcellation as hp
from nipy.labs.spatial_models.parcel_io import parcel_input, \
                                    write_parcellation_images

DATADIR = '/home/jb/.local/share/nipy/nipy/data/imagen/mid_task_contrast10'

def usage(msg):
    print(msg)
    quit()

def test_same_shapes(imgs):
    """ Test if the list of images have the same shape
    """
    for img in imgs:
        if (img.shape != imgs[0].shape):
            print img.shape, imgs[0].shape
            return False
    return True

def rm_indices_in(alist,indices):
    """ Remove the elements of alist at the given indices

    Parameters
    ----------
    indices: list
        contains the indices to be removed from a list
    alist: list
        of which elements are removed

    Returns
    -------
    alist: list

    Examples
    --------
    >>> alist = ['toto', 1, 2, 17, 'r']
    >>> indices = [2,1]
    >>> print rm_indices_in(alist,indices)
    ['toto', 17, 'r']
    """

    indice = list(set(indices)) # remove duplicates
    indices.sort()              # sort and reverse to get last indices first
    indices.reverse()
    for i in indices:           # start with the last indice
        alist.pop(i)            # rm towards the begining of the list
    return alist

#------------------------------------------------------------------------------
# step 1: get the data :
#------------------------------------------------------------------------------

imgfiles = [os.path.join(DATADIR,f) for f in os.listdir(DATADIR)]
n_subj = len(imgfiles)
if n_subj < 3:
    usage('too few subjects to work with')

imgs = [nib.load(f) for f in imgfiles]
if not test_same_shapes(imgs):
    usage('images should all have the same dimension')

# create tmp dir to store the masks
tmpdir = mkdtemp()

img_to_rm = []
masks = []
newimgs = []

for i,img in enumerate(imgs):
    # create a mask from the NANs (contrast or t images from SPM output)
    # also create images with NANs replaced by zeros ?
    fname = imgfiles[i]
    data = img.get_data()
    imghdr = img.get_header()
    mask = np.zeros(img.shape, dtype=data.dtype)
    mask[~np.isnan(data)] = 1
    new_data = data.copy()
    new_data[np.isnan(data)] = 0

    #------- mask and new img filename in temporary directory
    base,ext = os.path.splitext(fname)
    _,fbase = os.path.split(base)
    maskname = os.path.join(tmpdir, fbase+'_mask'+ext)
    nimgname = os.path.join(tmpdir, fbase+'_nonans'+ext)
    masks = masks + [maskname]
    newimgs = newimgs + [nimgname]

    #------- write mask and new image
    maskimage = nib.Nifti1Image(mask,img.get_affine(),header=imghdr)
    newcimage = nib.Nifti1Image(new_data,img.get_affine(),header=imghdr)
    maskimage.to_filename(maskname)
    newcimage.to_filename(nimgname)

    #------- imagen related issue: make sure the contrast was estimable
    if imghdr['descrip'].tostring().find('unestimable') >= 0:
        print('Damned %s' % fname);
        img_to_rm = img_to_rm + [i]

# remove from lists the non estimable contrasts
# use set and sort the lists
for i in img_to_rm:
    print('not using %s ' % newimgs[i])

masks = rm_indices_in(masks, img_to_rm)
newimgs = rm_indices_in(newimgs, img_to_rm)
n_subj = len(newimgs)

#------------------------------------------------------------------------------
# step 2 : prepare the parcel structure
#------------------------------------------------------------------------------

# parameter for the intersection of the mask
ths = .5

# possibly, dimension reduction can performed on the input data
# (not recommended)
fdim = 3

limgfiles = [[f] for f in newimgs]
domain, ldata = parcel_input(masks, limgfiles, ths, fdim)

#------------------------------------------------------------------------------
# step 3 : run the algorithm
#------------------------------------------------------------------------------

# number of parcels
nbparcel = 100
Pa = hp.hparcel(domain, ldata, nbparcel, mu=3.0, verbose=1)
# note: play with mu to change the 'stiffness of the parcellation'

# produce some output images: list of image filename as subject identifiers:
bases = [os.path.basename(img) for img in newimgs]
bnoext = [os.path.splitext(oneb)[0] for oneb in bases]

# The above should produce something like :
# ['000015734458_con_0010_nonans',
# ....
#  '000014900981_con_0010_nonans']

# write parcellation images in tmpdir
write_parcellation_images(Pa, subject_id=bnoext, swd=tmpdir, msg='cont 10 hp')

#------------------------------------------------------------------------------
# step 4:  look at the results
#------------------------------------------------------------------------------

import matplotlib.pylab as mp
# choose an axial slice in percentage

fparcels = [os.path.join(tmpdir, "parcel%s.nii" % bnoext[s])
                        for s in range(n_subj)]

ax_pc = .7
for s in range(min(n_subj,15)):
    mp.figure(figsize=(12, 5))
    img = nib.load(newimgs[s])
    data = img.get_data()
    mp.title('Input data subject %d' % s )
    mp.subplot(121)
    mp.imshow(data[:,:,int(data.shape[2]*ax_pc)], interpolation='nearest')

    imgparcel = nib.load(fparcels[s])
    data = imgparcel.get_data()
    mp.title('Parcel data subject %d' % s)
    mp.subplot(122)
    mp.imshow(data[:,:,int(data.shape[2]*ax_pc)], interpolation='nearest',
              vmin=-1, vmax=nbparcel)


