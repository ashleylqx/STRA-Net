##############
# PARAMETERS #
#################################
# model settings
model_no = 1 # 0: FeatureNet only ; 1: Whole network

if model_no == 0:
    FeatureNet_mode = 'main'
elif model_no == 1:
    FeatureNet_mode = 'feature'

phase = 'vis' # 'train' for training, 'vis' for visualization/predicting


# batch size
video_b_s = 1
image_b_s = 10

b_s = 10

# number of frames
num_frames = 5

# number of epochs
nb_epoch = 100

# number of timestep
nb_timestep = 4

# number of learned priors
nb_gaussian = 16

img_channel_mean = [103.939, 116.779, 123.68]

shape_c = 224
shape_r = 224

shape_c_out = 224
shape_r_out = 224

shape_c_gaus = 28
shape_r_gaus = 28

# number of rows of attention
shape_r_attention = shape_r_gaus#60
# number of cols of attention
shape_c_attention = shape_r_gaus

opt_num = 5
f_gap = 2

opt_small = 0.1
opt_large = 20

###################
# TRAINING CONFIG #
#################################
work_path = '/research/dept2/qxlai/'

model_path = '/research/dept2/qxlai/'

# DataSets paths
Linux_path = '/research/dept2/qxlai/DataSets/'

# DataSets = ['DHF1K'] # training setting 1
# DataSets = ['Hollywood-2'] # training setting 2
DataSets = ['UCF sports'] # training setting 3
# DataSets = ['UCF sports', 'Hollywood-2', 'DHF1K'] # training setting 4
# DataSets = ['DIEM_TE'] # testing only

videos_train_paths = [Linux_path+ds+'/train/' for ds in DataSets]
videos_val_paths = [Linux_path+ds+'/test/' for ds in DataSets]
videos_test_paths = [Linux_path+ds+'/test/' for ds in DataSets] # 'test' for three/'eval' for dhf1k
# path of  maps
maps_path = '/maps/'
# path of fixations 
fixs_path = '/fixation/'
# path of images
frames_path = '/images/'
# path of optical flows
optflw_path = '/flow/'

train_samples = {'UCF sports':103, 'Hollywood-2':3223, 'DHF1K': 600, 'ALL':3926, 'DIEM_TE':0}
test_samples = {'UCF sports':47, 'Hollywood-2':3701, 'DHF1K': 100, 'ALL':3848, 'DIEM_TE':20}

# number of training video
nb_train = train_samples[DataSets[0]]*1
# number of validation video
nb_videos_val = test_samples[DataSets[0]]*1
