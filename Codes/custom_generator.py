import os
import random
import numpy as np
from utilities import preprocess_maps, preprocess_fixmaps_png
from utilities import preprocess_images, preprocess_optflws_multi

from config import *


def generator_td_simple_mopt(phase_gen='train'):
    if phase_gen == 'train':
        videos_paths = videos_train_paths
    elif phase_gen == 'test':
        videos_paths = videos_val_paths
    else:
        raise NotImplementedError

    videos = [videos_path + f for videos_path in videos_paths for f in
              os.listdir(videos_path) if os.path.isdir(videos_path + f)]

    random.shuffle(videos)

    video_counter = 0
    while True:
        Xims_ops = np.zeros((num_frames, shape_r, shape_c, 3 + 2 * opt_num))
        # Xops = np.zeros((video_b_s, num_frames, shape_r, shape_c, 2))

        Ymaps = np.zeros((num_frames, shape_r_gaus, shape_c_gaus, 1)) + 0.01
        Yfixs = np.zeros((num_frames, shape_r_gaus, shape_c_gaus, 1)) + 0.01

        for i in range(0, video_b_s):
            video_path = videos[(video_counter + i) % len(videos)]
            images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                      f.endswith(('.jpg', '.jpeg', '.png'))]
            images.sort()

            
            optflws = [video_path + optflw_path + f for f in os.listdir(video_path + optflw_path) if
                       f.endswith('.flo')]

            optflws.sort()

            if len(optflws) < 10:
                continue

            for j in range(0, opt_num + 1):
                optflws.append(optflws[-1])

            maps = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if
                    f.endswith(('.jpg', '.jpeg', '.png'))]
            maps.sort()

            fixs = [video_path + fixs_path + f for f in os.listdir(video_path + fixs_path) if
                    f.endswith(('.jpg', '.jpeg', '.png'))]
            fixs.sort()

            start = random.randint(0, max(len(images) - num_frames * f_gap, 0))
            X = preprocess_images(images[start:min(start + num_frames * f_gap, len(images)):f_gap], shape_r, shape_c)
            X_f = preprocess_optflws_multi(optflws[start:min(start + num_frames * f_gap + opt_num, len(optflws))],
                                           shape_r, shape_c, X.shape[0])  # how to get filename list????
            Y = preprocess_maps(maps[start:min(start + num_frames * f_gap, len(images)):f_gap], shape_r_gaus,
                                shape_c_gaus)
            Y_fix = preprocess_fixmaps_png(fixs[start:min(start + num_frames, len(images)):f_gap], shape_r_gaus,
                                           shape_c_gaus)

            # print('Xims_ops:', Xims_ops.shape)
            # print('X:', X.shape)
            # print('X_f:', X_f.shape)
            Xims_ops[0:X.shape[0], :, :, 0:3] = np.copy(X)
            Xims_ops[0:X_f.shape[0], :, :, 3:] = np.copy(X_f)
            Ymaps[0:Y.shape[0], :] = np.copy(Y)
            Yfixs[0:Y_fix.shape[0], :] = np.copy(Y_fix)
            
            Xims_ops[X.shape[0]:num_frames, :, :, 0:3] = np.copy(X[-1, :, :])
            Xims_ops[X_f.shape[0]:num_frames, :, :, 3:] = np.copy(X_f[-1, :, :])
            Ymaps[Y.shape[0]:num_frames, :] = np.copy(Y[-1, :, :])
            Yfixs[Y_fix.shape[0]:num_frames, :] = np.copy(Y_fix[-1, :, :])

        yield Xims_ops, \
              [Ymaps, Ymaps, Yfixs, Ymaps,
               Ymaps, Ymaps, Yfixs, Ymaps,
               Ymaps, Ymaps, Yfixs, Ymaps,
               Ymaps, Ymaps, Yfixs, Ymaps
               ]  #
        video_counter = (video_counter + video_b_s) % len(videos)


def generator_td_prior_mopt_ds(phase_gen='train'):
    if phase_gen == 'train':
        videos_paths = videos_train_paths
    elif phase_gen == 'test':
        videos_paths = videos_val_paths
    else:
        raise NotImplementedError

    videos = [videos_path + f for videos_path in videos_paths for f in
              os.listdir(videos_path) if os.path.isdir(videos_path + f)]

    random.shuffle(videos)

    video_counter = 0
    while True:
        Xims_ops = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3 + 2 * opt_num))
        # Xops = np.zeros((video_b_s, num_frames, shape_r, shape_c, 2))

        Ymaps = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1))
        Yfixs = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1))

        gaussian = np.zeros((video_b_s, shape_r_gaus, shape_c_gaus, nb_gaussian))
        gaussians = [gaussian for i in range(0, num_frames)]

        for i in range(0, video_b_s):
            video_path = videos[(video_counter + i) % len(videos)]
            images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                      f.endswith(('.jpg', '.jpeg', '.png'))]
            images.sort()

            optflws = [video_path + optflw_path + f for f in os.listdir(video_path + optflw_path) if
                       f.endswith('.flo')]

            optflws.sort()

            if len(optflws) < 10:
                continue

            for j in range(0, opt_num + 1):
                optflws.append(optflws[-1])

            maps = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if
                    f.endswith(('.jpg', '.jpeg', '.png'))]
            maps.sort()

            fixs = [video_path + fixs_path + f for f in os.listdir(video_path + fixs_path) if
                    f.endswith(('.jpg', '.jpeg', '.png'))]
            fixs.sort()

            start = random.randint(0, max(len(images) - num_frames * f_gap, 0))
            X = preprocess_images(images[start:min(start + num_frames * f_gap, len(images)):f_gap], shape_r, shape_c)
            X_f = preprocess_optflws_multi(optflws[start:min(start + num_frames * f_gap + opt_num, len(optflws))],
                                           shape_r, shape_c, X.shape[0])  # how to get filename list????
            Y = preprocess_maps(maps[start:min(start + num_frames * f_gap, len(images)):f_gap], shape_r_out,
                                shape_c_out)
            Y_fix = preprocess_fixmaps_png(fixs[start:min(start + num_frames, len(images)):f_gap], shape_r_out,
                                           shape_c_out)

            Xims_ops[i, 0:X.shape[0], :, :, 0:3] = np.copy(X)
            Xims_ops[i, 0:X_f.shape[0], :, :, 3:] = np.copy(X_f)
            Ymaps[i, 0:Y.shape[0], :] = np.copy(Y)
            Yfixs[i, 0:Y_fix.shape[0], :] = np.copy(Y_fix)

            Xims_ops[i, X.shape[0]:num_frames, :, :, 0:3] = np.copy(X[-1, :, :])
            Xims_ops[i, X_f.shape[0]:num_frames, :, :, 3:] = np.copy(X_f[-1, :, :])
            Ymaps[i, Y.shape[0]:num_frames, :] = np.copy(Y[-1, :, :])
            Yfixs[i, Y_fix.shape[0]:num_frames, :] = np.copy(Y_fix[-1, :, :])

        gaussians.append(Xims_ops)
        yield gaussians, \
              [Ymaps, Ymaps, Yfixs, Ymaps,
               Ymaps, Ymaps, Yfixs, Ymaps,
               Ymaps, Ymaps, Yfixs, Ymaps,
               Ymaps, Ymaps, Yfixs, Ymaps
               ]  #
        video_counter = (video_counter + video_b_s) % len(videos)


def get_test_td_simple_mopt(video_test_path):
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]

    optflws = [video_test_path + optflw_path + f for f in os.listdir(video_test_path + optflw_path) if
               f.endswith(('.flo\n'))]

    # add one more optical flow
    for j in range(0, opt_num + 1):
        optflws.append(optflws[-1])

    images.sort()
    optflws.sort()
    start = 0

    # true???? note the val steps when using predictor
    while True:
        Xims_ops = np.zeros((num_frames, shape_r, shape_c, 3 + 2 * opt_num))
        # Xops = np.zeros((1, num_frames, shape_r, shape_c, 2))

        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        X_f = preprocess_optflws_multi(optflws[start:min(start + num_frames + opt_num, len(optflws))], shape_r, shape_c,
                                       X.shape[0], phase='test')

        Xims_ops[0:min(len(images) - start, num_frames), :, :, 0:3] = np.copy(X)
        Xims_ops[0:min(len(images) - start, num_frames), :, :, 3:] = np.copy(X_f)

        yield Xims_ops  #
        start = min(start + num_frames, len(images))


def get_test_td_prior_mopt(video_test_path):
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]

    optflws = [video_test_path + optflw_path + f for f in os.listdir(video_test_path + optflw_path) if
               f.endswith(('.flo'))]

    # add one more optical flow
    images.sort()
    optflws.sort()
        
    for j in range(0, opt_num+1):
        optflws.append(optflws[-1])

    start = 0
    
    while True:
        Xims_ops = np.zeros((1, num_frames, shape_r, shape_c, 3+2*opt_num))
        gaussian = np.zeros((1, shape_r_gaus, shape_c_gaus, nb_gaussian))
        data = [gaussian for i in range(0, num_frames)]
        
        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        X_f = preprocess_optflws_multi(optflws[start:min(start + num_frames+opt_num+1, len(optflws))], shape_r, shape_c, X.shape[0], phase='test')
        
        Xims_ops[0, 0:min(len(images)-start, num_frames), :,:, 0:3] = np.copy(X)
        Xims_ops[0, 0:min(len(images)-start, num_frames), :,:, 3:] = np.copy(X_f)
        
        data.append(Xims_ops)
        yield data #
        start = min(start + num_frames, len(images))
