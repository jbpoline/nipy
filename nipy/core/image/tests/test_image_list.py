# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ..image_list import ImageList
from ..image import Image
from ....io.api import load_image

from ....testing import (funcfile, assert_true, assert_equal, assert_raises,
                        assert_almost_equal)

def test_image_list():
    img = load_image(funcfile)
    exp_shape = (17, 21, 3, 20)
    imglst = ImageList.from_image(img, axis=-1)
    # Test empty ImageList
    emplst = ImageList()
    assert_equal(len(emplst.list), 0)

    # Test non-image construction
    a = np.arange(10)
    assert_raises(ValueError, ImageList, a)

    # Test axis must be specified
    assert_raises(ValueError, ImageList.from_image, img)
    assert_raises(ValueError, ImageList.from_image, img, None)

    # check all the axes
    for i in range(4):
        order = range(4)
        order.remove(i)
        order.insert(0,i)
        img_re_i = img.reordered_reference(order).reordered_axes(order)
        imglst_i = ImageList.from_image(img, axis=i)
        assert_equal(imglst_i.list[0].shape, img_re_i.shape[1:])
        # check the affine as well
        assert_almost_equal(imglst_i.list[0].affine, img_re_i.affine[1:,1:])

    assert_equal(img.shape, exp_shape)

    # length of image list should match number of frames
    assert_equal(len(imglst.list), img.shape[3])
    # check the affine
    A = np.identity(4)
    A[:3,:3] = img.affine[:3,:3]
    A[:3,-1] = img.affine[:3,-1]
    assert_almost_equal(imglst.list[0].affine, A)

    # Slicing an ImageList should return an ImageList
    sublist = imglst[2:5]
    assert_true(isinstance(sublist, ImageList))
    # Except when we're indexing one element
    assert_true(isinstance(imglst[0], Image))
    # Verify array interface
    # test __array__
    assert_true(isinstance(sublist.get_data(axis=0), np.ndarray))
    # Test __setitem__
    sublist[2] = sublist[0]
    assert_equal(sublist[0].get_data().mean(),
                 sublist[2].get_data().mean())
    # Test iterator
    for x in sublist:
        assert_true(isinstance(x, Image))
        assert_equal(x.shape, exp_shape[:3])

    # Test image_list.get_data(axis = an_axis)
    funcim = load_image(funcfile)
    ilist = ImageList.from_image(funcim, axis='t')

    # make sure that we pass an axis
    assert_raises(ValueError, ImageList.get_data, ilist, None)
    assert_raises(ValueError, ImageList.get_data, ilist)

    # make sure that axis that dont exist makes the function fails
    assert_raises(ValueError, ImageList.get_data, ilist, 4)
    assert_raises(ValueError, ImageList.get_data, ilist, -5)

    # make sure that axis is put in the right place in the result array
    # image of ilist have dimension (17,21,3), lenght(ilist) = 20.
    data = ilist.get_data(axis='first')
    assert_equal(data.shape, (20, 17, 21, 3))

    data = ilist.get_data(axis=0)
    assert_equal(data.shape, (20, 17, 21, 3))

    data = ilist.get_data(axis=1)
    assert_equal(data.shape, (17, 20, 21, 3))

    data = ilist.get_data(axis=2)
    assert_equal(data.shape, (17, 21, 20, 3))

    data = ilist.get_data(axis=3)
    assert_equal(data.shape, (17, 21, 3, 20))

    data = ilist.get_data(axis=-1)
    assert_equal(data.shape, (17, 21, 3, 20))

    data = ilist.get_data(axis='last')
    assert_equal(data.shape, (17, 21, 3, 20))

    data = ilist.get_data(axis=-2)
    assert_equal(data.shape, (17, 21, 20, 3))

