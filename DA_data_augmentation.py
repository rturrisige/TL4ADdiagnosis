"""
This code generates an augmented dataset based on affine transformations.
Specifically, given a 3D MRI image zoom, shift and rotation are separately applied, providing 3 augmented images.
"""

import numpy as np
from scipy import ndimage
import argparse
from glob import glob as gg
import os
# Define functions


def add_noise(x):
    s = np.std(x)/10
    noise = np.random.normal(loc=0, scale=s, size=x.shape)
    return x + noise


def random_zoom(matrix, min_percentage=0.8, max_percentage=1.2):
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, zoom_matrix)


def random_rotate3D(img_numpy, min_angle, max_angle):
    """
   Returns a random rotated array in the same shape
   :param img_numpy: 3D numpy array
   :param min_angle: in degrees
   :param max_angle: in degrees
   """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, max_percentage=0.4):
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


def img_processing(data_dir, saver_dir, n_augmentations=1):
    if not os.path.exists(saver_dir + '/affine'):
        os.makedirs(saver_dir + '/affine')
    if not os.path.exists(saver_dir + '/noise'):
        os.makedirs(saver_dir + '/noise')
    files = gg(data_dir + '/*.npy')
    i = 0
    for f in files:
        data = np.load(f, allow_pickle=True)
        if len(data) == 2:
            input_image = data[0]
        else:
            input_image = data
        name = os.path.basename(f).split('.npy')[0]
        for n in range(n_augmentations):
            zoom = random_zoom(input_image, min_percentage=0.8, max_percentage=1.2)
            shift = random_shift(input_image)
            rotation = random_rotate3D(input_image, min_angle=-5, max_angle=5)
            gauss_noise = add_noise(input_image)
            np.save(saver_dir + '/affine/' + name + 'zoom_' + str(n+1) + '.npy', zoom)
            np.save(saver_dir + '/affine/' + name + 'shift_' + str(n+1) + '.npy', shift)
            np.save(saver_dir + '/affine/' + name + 'rotation_' + str(n+1) + '.npy', rotation)
            np.save(saver_dir + '/noise/' + name + '_gaussian_noise_' + str(n+1) + '.npy', gauss_noise)
        i += 1
        if i % 50 == 0:
            print(i, 'files processed.')

##


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Create augmented dataset from <data_dir> 
    by applying zoom, shift and rotation. Augmented data are saved in
    <saver_dir>.""")
    parser.add_argument('--data_dir', required=True, type=str,
                        help='The directory that contains npy files to be processed')
    parser.add_argument('--saver_dir', default=os.getcwd(),
                        type=str, help='The directory where to save the augmented data')
    parser.add_argument('--n_augmentation', default=1, type=int,
                        help='Number of times to apply each transformation')
    args = parser.parse_args()
    img_processing(args.data_dir, args.saver_dir, n_augmentations=args.n_augmentation)
