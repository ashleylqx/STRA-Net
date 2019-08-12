from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
# from models.BilinearUpSampling import BilinearUpSampling2D
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.47
sess = tf.Session(config=config)
'''
import numpy as np

from keras.layers import merge, Input, Activation, add, multiply
from keras.layers import Flatten, RepeatVector, Permute, Softmax

from keras.layers import Reshape
from keras. models import Model

from keras.layers import Lambda
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed

from config import *
from gaussian_prior import gaussian_priors_init, LearningPrior
from BilinearUpSampling import BilinearUpSampling2D
from convgru import ConvGRU2D

from utilities import postprocess_predictions, postprocess_predictions_2
from math import ceil

from scipy.misc import imread, imsave

from custom_generator import get_test_td_prior_mopt


def Slice_inputs(input):
    Ximgs = input[:,:,:,0:3]
    Xopts = input[:,:,:,3:]
    return [Ximgs, Xopts]

def Slice_output_shape(input_shape):
    return [input_shape[0:3]+(3,), input_shape[0:3]+(2*opt_num,)]
    
def Expand_gaus(input):
    return K.expand_dims(input, axis=-4)    

def resize_like(input_tensor, ref_tensor): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value])    

##### slice the features and aux outputs
def Slice_outputs_mask(input):
    frame_features = input[:,:,:,:,0:2048]
    aux_out1 = input[:,:,:,:,2048:2049]
    aux_out2 = input[:,:,:,:,2049:2050]
    aux_out3 = input[:,:,:,:,2050:2051]
    mask3 = input[:,:,:,:,2051:2052]
    mask4 = input[:,:,:,:,2052:2053]
    mask5 = input[:,:,:,:,2053:]
    
    return [frame_features, aux_out1, aux_out2, aux_out3, mask3, mask4, mask5]

def Slice_outs_shape_mask(input_shape):
    return [input_shape[0:4]+(2048,), input_shape[0:4]+(1,), input_shape[0:4]+(1,), input_shape[0:4]+(1,),
            input_shape[0:4]+(1,), input_shape[0:4]+(1,), input_shape[0:4]+(1,)]

#### resnet blocks ####
def identity_block(input_tensor, kernel_size, filters, stage, block, stream, TRAINABLE):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = stream + '_res' + str(stage) + block + '_branch'
    bn_name_base = stream + '_bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=TRAINABLE)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      padding='same', name=conv_name_base + '2b', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # x = merge([x, input_tensor], mode='sum')
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, stream, TRAINABLE, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = stream + '_res' + str(stage) + block + '_branch'
    bn_name_base = stream + '_bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', trainable=TRAINABLE)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', trainable=TRAINABLE)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # x = merge([x, shortcut], mode='sum')
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def conv_block_atrous(input_tensor, kernel_size, filters, stage, block, stream, TRAINABLE, atrous_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = stream + '_res' + str(stage) + block + '_branch'
    bn_name_base = stream + '_bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=TRAINABLE)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               dilation_rate=atrous_rate, name=conv_name_base + '2b', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '1', trainable=TRAINABLE)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # x = merge([x, shortcut], mode='sum')
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block_atrous(input_tensor, kernel_size, filters, stage, block, stream, TRAINABLE, atrous_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = stream + '_res' + str(stage) + block + '_branch'
    bn_name_base = stream + '_bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=TRAINABLE)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
               padding='same', name=conv_name_base + '2b', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=TRAINABLE)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # x = merge([x, input_tensor], mode='sum')
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x    

# attention module
def att(input_feature):
    chnn = (K.int_shape(input_feature)[-1])//2
    out = Conv2D(chnn, (3, 3), padding = 'same', activation='relu')(input_feature)
    out = BatchNormalization()(out)
    out = Conv2D(1, (1, 1), padding = 'same', activation='sigmoid', kernel_initializer='ones')(out)    
    return out        

def att_f(input_att, chnn):
    f_att = Flatten()(input_att)
    f_att = Softmax()(f_att)
    f_att = RepeatVector(chnn)(f_att)
    # print('before', K.int_shape(f_att))
    f_att = Permute((2, 1))(f_att)
    # print('after', K.int_shape(f_att))
    f_att = Reshape((shape_r_attention, shape_c_attention, chnn))(f_att)
    # print('final', K.int_shape(f_att))
    return f_att    

def att_mask(input_att):
    f_att = Flatten()(input_att)
    f_att = Softmax()(f_att)
    f_att = Reshape((shape_r_attention, shape_c_attention, 1))(f_att)
    
    return f_att

### This version CAN visualize masks #####
def Feature_dcross_res_matt_res_ds_masks(input_tensors=None, a_train=True, m_train=True):
    if input_tensors is None:
        input_tensors = Input(shape=(shape_r, shape_c, 3+2*opt_num))        

    Ximgs, Xopts = Lambda(Slice_inputs, output_shape=Slice_output_shape)(input_tensors)
    #print('Ximgs:', K.int_shape(Ximgs))
    #print('Xopts:', K.int_shape(Xopts))
    
    bn_axis = 3
    TRAINABLE_A = a_train
    TRAINABLE_M = m_train    
    
    ####### Motion stream ########
    stream = 'M'
    
    # conv_1
    M_conv_1_out = Conv2D(64, (7, 7), strides=(2, 2), name='M_conv1', trainable=TRAINABLE_M)(Xopts) # 112*112
    M_conv_1_out = BatchNormalization(axis=bn_axis, name='M_bn_conv1')(M_conv_1_out)
    M_conv_1_out = Activation('relu')(M_conv_1_out)
    M_ds_conv_1_out = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='M_maxpooling')(M_conv_1_out) # 56*56

    # conv_2
    M_conv_2_out = conv_block(M_ds_conv_1_out, 3, [64, 64, 256], stage=2, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_M, strides=(1, 1))
    M_conv_2_out = identity_block(M_conv_2_out, 3, [64, 64, 256], stage=2, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    M_conv_2_out = identity_block(M_conv_2_out, 3, [64, 64, 256], stage=2, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_M)

    # conv_3
    M_conv_3_out = conv_block(M_conv_2_out, 3, [128, 128, 512], stage=3, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_M, strides=(2, 2))# 28*28
    M_conv_3_out = identity_block(M_conv_3_out, 3, [128, 128, 512], stage=3, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    M_conv_3_out = identity_block(M_conv_3_out, 3, [128, 128, 512], stage=3, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    M_conv_3_out = identity_block(M_conv_3_out, 3, [128, 128, 512], stage=3, block='d',
                            stream = stream, TRAINABLE = TRAINABLE_M)

    # conv_4
    M_conv_4_out = conv_block_atrous(M_conv_3_out, 3, [256, 256, 1024], stage=4, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_M) 
    M_conv_4_out = identity_block_atrous(M_conv_4_out, 3, [256, 256, 1024], stage=4, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    M_conv_4_out = identity_block_atrous(M_conv_4_out, 3, [256, 256, 1024], stage=4, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    M_conv_4_out = identity_block_atrous(M_conv_4_out, 3, [256, 256, 1024], stage=4, block='d',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    M_conv_4_out = identity_block_atrous(M_conv_4_out, 3, [256, 256, 1024], stage=4, block='e',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    M_conv_4_out = identity_block_atrous(M_conv_4_out, 3, [256, 256, 1024], stage=4, block='f',
                            stream = stream, TRAINABLE = TRAINABLE_M)
    
    # conv_5
    M_conv_5_out = conv_block_atrous(M_conv_4_out, 3, [512, 512, 2048], stage=5, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_M, atrous_rate=(4, 4))
    M_conv_5_out = identity_block_atrous(M_conv_5_out, 3, [512, 512, 2048], stage=5, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_M, atrous_rate=(4, 4))
    M_conv_5_out = identity_block_atrous(M_conv_5_out, 3, [512, 512, 2048], stage=5, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_M, atrous_rate=(4, 4))
        
    ####### Apprearence stream ########
    stream = 'A'
    
    # conv_1
    A_conv_1_out = Conv2D(64, (7, 7), strides=(2, 2), name='A_conv1', trainable=TRAINABLE_A)(Ximgs) # 112*112
    A_conv_1_out = BatchNormalization(axis=bn_axis, name='A_bn_conv1')(A_conv_1_out)
    A_conv_1_out = Activation('relu')(A_conv_1_out)
    A_ds_conv_1_out = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name = 'A_maxpooling')(A_conv_1_out) # 56*56

    # conv_2
    A_conv_2_out = conv_block(A_ds_conv_1_out, 3, [64, 64, 256], stage=2, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_A, strides=(1, 1))
    A_conv_2_out = identity_block(A_conv_2_out, 3, [64, 64, 256], stage=2, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    A_conv_2_out = identity_block(A_conv_2_out, 3, [64, 64, 256], stage=2, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_A)

    # conv_3
    A_conv_3_out = conv_block(A_conv_2_out, 3, [128, 128, 512], stage=3, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_A, strides=(2, 2))# 28*28
    A_conv_3_out = identity_block(A_conv_3_out, 3, [128, 128, 512], stage=3, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    A_conv_3_out = identity_block(A_conv_3_out, 3, [128, 128, 512], stage=3, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    A_conv_3_out = identity_block(A_conv_3_out, 3, [128, 128, 512], stage=3, block='d',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    
    A_conv_3_out_M = multiply([A_conv_3_out, M_conv_3_out])
    A_conv_3_out_M = add([A_conv_3_out, A_conv_3_out_M])
    
    AM_att_3 = att(A_conv_3_out_M)
    f_AM_att_3 = att_f(AM_att_3, 512)
    
    A_conv_3_out_M2 = multiply([A_conv_3_out_M, f_AM_att_3])
    A_conv_3_out_M2 = add([A_conv_3_out_M, A_conv_3_out_M2])

    # conv_4
    A_conv_4_out = conv_block_atrous(A_conv_3_out_M2, 3, [256, 256, 1024], stage=4, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_A) 
    A_conv_4_out = identity_block_atrous(A_conv_4_out, 3, [256, 256, 1024], stage=4, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    A_conv_4_out = identity_block_atrous(A_conv_4_out, 3, [256, 256, 1024], stage=4, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    A_conv_4_out = identity_block_atrous(A_conv_4_out, 3, [256, 256, 1024], stage=4, block='d',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    A_conv_4_out = identity_block_atrous(A_conv_4_out, 3, [256, 256, 1024], stage=4, block='e',
                            stream = stream, TRAINABLE = TRAINABLE_A)
    A_conv_4_out = identity_block_atrous(A_conv_4_out, 3, [256, 256, 1024], stage=4, block='f',
                            stream = stream, TRAINABLE = TRAINABLE_A)
        
    A_conv_4_out_M = Concatenate()([M_conv_4_out, M_conv_3_out])
    A_conv_4_out_M = Conv2D(1024, (3,3), activation='relu', padding='same')(A_conv_4_out_M)
    A_conv_4_out_M = multiply([A_conv_4_out_M, A_conv_4_out])
    A_conv_4_out_M = add([A_conv_4_out, A_conv_4_out_M])
    
    AM_att_4 = att(A_conv_4_out_M)
    f_AM_att_4 = att_f(AM_att_4, 1024)
    
    A_conv_4_out_M2 = multiply([A_conv_4_out_M, f_AM_att_4])
    A_conv_4_out_M2 = add([A_conv_4_out_M, A_conv_4_out_M2])

    # conv_5
    A_conv_5_out = conv_block_atrous(A_conv_4_out_M2, 3, [512, 512, 2048], stage=5, block='a',
                            stream = stream, TRAINABLE = TRAINABLE_A, atrous_rate=(4, 4))
    A_conv_5_out = identity_block_atrous(A_conv_5_out, 3, [512, 512, 2048], stage=5, block='b',
                            stream = stream, TRAINABLE = TRAINABLE_A, atrous_rate=(4, 4))
    A_conv_5_out = identity_block_atrous(A_conv_5_out, 3, [512, 512, 2048], stage=5, block='c',
                            stream = stream, TRAINABLE = TRAINABLE_A, atrous_rate=(4, 4))
    

    A_conv_5_out_M = Concatenate()([M_conv_5_out, M_conv_4_out, M_conv_3_out])
    A_conv_5_out_M = Conv2D(2048, (3,3), activation='relu', padding='same')(A_conv_5_out_M)
    A_conv_5_out_M = multiply([A_conv_5_out_M, A_conv_5_out])
    A_conv_5_out_M = add([A_conv_5_out, A_conv_5_out_M])

    AM_att_5 = att(A_conv_5_out_M)
    f_AM_att_5 = att_f(AM_att_5, 2048)
    
    A_conv_5_out_M2 = multiply([A_conv_5_out_M, f_AM_att_5])
    A_conv_5_out_M2 = add([A_conv_5_out_M, A_conv_5_out_M2])
    
    
    # side outputs
    sd_3 = Conv2D(1, (1,1), padding='same', activation='sigmoid')(A_conv_3_out_M2)
    sd_4 = Conv2D(1, (1,1), padding='same', activation='sigmoid')(A_conv_4_out_M2)
    sd_5 = Conv2D(1, (1,1), padding='same', activation='sigmoid')(A_conv_5_out_M2)

    # mask outputs
    m3 = att_mask(AM_att_3)
    m4 = att_mask(AM_att_4)
    m5 = att_mask(AM_att_5)
    
    # Create model
    model = Model(input_tensors,
              Concatenate()([A_conv_5_out_M2, sd_3, sd_4, sd_5, m3, m4, m5]))

    return model

def TD_model_prior_masks(input_tensors=None, f1_train=True, stateful=False):
    f1 = Feature_dcross_res_matt_res_ds_masks()
    f1.trainable = f1_train
    
    if input_tensors is None:
        xgaus_shape = (shape_r_gaus, shape_c_gaus, nb_gaussian)
        ximgs_ops_shape = (None, shape_r, shape_c, 3+2*opt_num)
        input_tensors = [Input(shape=xgaus_shape) for i in range(0,num_frames)]
        input_tensors.append(Input(shape=ximgs_ops_shape))

    Ximgs_ops = input_tensors[-1] 
    Xgaus = input_tensors[:-1]
            
    features_out = TimeDistributed(f1)(Ximgs_ops)
    
    frame_features, aux_out1, aux_out2, aux_out3, mask3, mask4, mask5 \
        = Lambda(Slice_outputs_mask, output_shape=Slice_outs_shape_mask)(features_out)
    
    #print('frame_features', K.int_shape(frame_features))

    outs = ConvGRU2D(filters=256, kernel_size=(3, 3),
                   padding='same', return_sequences=True, stateful=stateful,
                   name='ConvGRU2D')(frame_features)

    outs = TimeDistributed(BatchNormalization(name='ConvGRU2D_BN'))(outs) # previously 256    
    
    prior_layer1 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)
    priors1 = [Lambda(Expand_gaus)(prior_layer1(x)) for x in Xgaus]
    priors1_merged = Concatenate(axis=-4)(priors1)
    
    sal_concat1 = Concatenate(axis=-1)([outs, priors1_merged])

    outs = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(sal_concat1)
    outs = TimeDistributed(BilinearUpSampling2D((8,8)))(outs)

    aux_out1 = TimeDistributed(BilinearUpSampling2D((8,8)))(aux_out1)
    aux_out2 = TimeDistributed(BilinearUpSampling2D((8,8)))(aux_out2)
    aux_out3 = TimeDistributed(BilinearUpSampling2D((8,8)))(aux_out3)

    # for visualization              
    model = Model(inputs=input_tensors,
              outputs=[outs,
                       aux_out1,
                       aux_out2,
                       aux_out3,
                       mask3,
                       mask4,
                       mask5
                       ],
              name = 'TD_model_prior')
    
    return model


if __name__ == '__main__':
    # TESTING
    stateful = True

    # get predictions only
    xgaus_bshape = (video_b_s, shape_r_gaus, shape_c_gaus, nb_gaussian)
    ximgs_ops_bshape = (video_b_s, num_frames, shape_r, shape_c, 3+2*opt_num)
    input_tensors = [Input(batch_shape=xgaus_bshape) for i in range(0,num_frames)]
    input_tensors.append(Input(batch_shape=ximgs_ops_bshape))

    m=TD_model_prior_masks(input_tensors, f1_train = True, stateful=stateful)


    weight_name = 'UHD_dcross_res_matt_res_p6_diem.01-299.6534'
    print("Loading %s.h5"%weight_name)
    m.load_weights(model_path+'vap_model/%s.h5'%weight_name)

    output = work_path+'vap_predictions/'+DataSets[0]+'/%s/'%weight_name

    videos = [videos_path + f for videos_path in videos_test_paths for f in
                  os.listdir(videos_path) if os.path.isdir(videos_path + f)]
    videos.sort()

    for i in range(len(videos)):
        print(videos[i])
        images_names = [f for f in os.listdir(videos[i] + frames_path) if
                  f.endswith(('.jpg', '.jpeg', '.png'))]
        images_names.sort()
        print('img_num:',len(images_names))
        # Output Folder Path
        if stateful:
            output_folder = output + '/' + os.path.split(videos[i])[1]+'/'
        else:
            output_folder = output + '/' + os.path.split(videos[i])[1]+'_not_stateful/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        predictions = m.predict_generator(get_test_td_prior_mopt(videos[i]),
                                         max(ceil(len(images_names)/num_frames),2))

        # only save the prediction maps
        for idx in [0]:
            prediction = predictions[idx]

            for j in range(len(images_names)):
                original_image = imread(videos[i] + frames_path + images_names[j])
                x, y = divmod(j, num_frames)

                res = prediction[x, y, :, :, 0]
                if idx<4:
                    res = postprocess_predictions(res, original_image.shape[0], original_image.shape[1])
                else:
                    res = postprocess_predictions_2(res, original_image.shape[0], original_image.shape[1])

                if idx==0:
                    imsave(output_folder + '/' + images_names[j], res)
                elif idx>3:
                    imsave(output_folder + '/' + images_names[j].replace('.png', '_%d.png'%(idx)), res)
                else:
                    side_folder = output + '_%d/'%(idx+2) + os.path.split(videos[i])[1]+'/'
                    if not os.path.exists(side_folder):
                        os.makedirs(side_folder)
                    imsave(side_folder + '/' + images_names[j], res)

        m.reset_states()
