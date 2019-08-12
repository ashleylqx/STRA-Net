from __future__ import division
import numpy as np
import scipy.io
import scipy.ndimage
from scipy.misc import imread, imsave, imresize
from config import *

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        # original_map = cv2.imread(path, 0)
        # original_map = mpimg.imread(path)
        original_map = imread(path)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, :, :, 0] = padded_map.astype(np.float32)
        ims[i, :, :, 0] /= 255.0

    return ims

def preprocess_fixmaps_png(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        fix_map = imread(path)
        ims[i, :, :, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims

def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]

        img = imresize(img, (shape_r, new_cols))
            
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]

        img = imresize(img, (new_rows,shape_c))
            
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        # original_image = cv2.imread(path)
        # original_image = mpimg.imread(path)
        original_image = imread(path)
        # if random.choice([0, 1]):
        #     original_image = imresize(original_image, (shape_r//2, shape_c//2))
        #     original_image = imresize(original_image, (shape_r, shape_c))
        
        if original_image.ndim == 2:
            copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image
            original_image = copy
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims[:, :, :, ::-1]
    # ims = ims.transpose((0, 3, 1, 2))

    return ims

def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        # pred = cv2.resize(pred, (new_cols, shape_r))
        pred = imresize(pred, (shape_r, new_cols))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        # pred = cv2.resize(pred, (shape_c, new_rows))
        pred = imresize(pred, (new_rows, shape_c))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r),:]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=7)
    #img = scipy.ndimage.filters.gaussian_filter(img, sigma=5)
    img = img / np.max(img) * 255

    return img

def postprocess_predictions_2(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = 1-pred
    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        # pred = cv2.resize(pred, (new_cols, shape_r))
        pred = imresize(pred, (shape_r, new_cols))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        # pred = cv2.resize(pred, (shape_c, new_rows))
        pred = imresize(pred, (new_rows, shape_c))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r),:]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=25)
    img = img / np.max(img) * 255

    return img

############### optical flow ##################################
def readFlow(name):
    # read .flo file
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def padding_optflws(img, shape_r=240, shape_c=320):
    img_padded = np.zeros((shape_r, shape_c, 2), dtype=np.float32)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]

        tmp = np.zeros((shape_r, new_cols, 2), dtype=np.float32)
        tmp[:,:,0] = np.resize(img[:,:,0], (shape_r, new_cols))
        tmp[:,:,1] = np.resize(img[:,:,1], (shape_r, new_cols))

        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = tmp
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]

        tmp = np.zeros((new_rows, shape_c, 2), dtype=np.float32)
        tmp[:,:,0] = np.resize(img[:,:,0], (new_rows, shape_c))
        tmp[:,:,1] = np.resize(img[:,:,1], (new_rows, shape_c))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = tmp

    # avoid noise

    idx = np.where(np.fabs(img_padded)<opt_small)
    img_padded[idx] = 0
    idx = np.where(img_padded>opt_large)
    img_padded[idx] = opt_large
    idx = np.where(img_padded<-opt_large)
    img_padded[idx] = -opt_large

    return img_padded

########## for multi opt and jump frames ###########
def preprocess_optflws_multi(paths, shape_r, shape_c, num_samples, phase='train'):
    ims = np.zeros((num_samples, shape_r, shape_c, 2*opt_num))
    # print(num_samples)
    # for i, path in enumerate(paths):
    if phase=='train':
        for i in range(0,num_samples,f_gap):
            for j in range(0, opt_num):
                original_flow = readFlow(paths[i+j])
                padded_flow = padding_optflws(original_flow, shape_r, shape_c)
                ims[i,:,:,2*j:(2*j+2)] = padded_flow
    else:
        for i in range(0,num_samples):
            for j in range(0, opt_num):
                # print('idx:', i+j)
                original_flow = readFlow(paths[i+j])
                padded_flow = padding_optflws(original_flow, shape_r, shape_c)
                ims[i,:,:,2*j:(2*j+2)] = padded_flow

    ims = ims[:, :, :, ::-1]
    return ims

