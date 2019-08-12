# STRA-Net
This repository is the implementation of **'Video Saliency Prediction using Spatiotemporal Residual Attentive Networks (TIP2019)'**.

Qiuxia Lai, [Wenguan Wang](https://sites.google.com/view/wenguanwang "wwg"), [Hanqiu Sun](http://www.cse.cuhk.edu.hk/~hanqiu/ "sunny"), Jianbing Shen

## Environment
- CentOS-7
- Python 3.5.2
- Tensorflow 1.11.0
- Keras 2.2.4
- CUDA 9.0
- CUDNN 7.5.0

## Results Download
Prediction results on **DHF1K**, **Hollywood-2**, **UCF sports**, and **DIEM** can be downloaded from:

Google Drive: <https://drive.google.com/file/d/1VmXVJ5H8y3-uihDrr1yTVPZBNIE0eoOW/view?usp=sharing>

Baidu Disk: <https://pan.baidu.com/s/1wvTtHuL5ra7umsG9_dICig>  password:dizw

## Preparation
### Generate optical flows
The optical flows are generated using [flownet-2.0](https://github.com/lmb-freiburg/flownet2 "flownet2"). The '**.flo**' files in the '**\flow**' folders under the video directory. Please be noted that the optical flow files may take up a considerable amout of storage space. The dataset directory is as follows: 
    
        DataSets
        |----DHF1K
             |---train
                 |--0001
                    |----images
                    |----flow
                    |----fixation
                    |----maps
                 |--0002
                    |----...
                 |--...
             |---test
                 |--...
        |----Hollowood-2
             |---train
             |---test
        |----UCF sports
             |---train
             |---test

For **Holloyood-2**, we further seperate the video sequences into shots according to the ground-truth shot boundaries, and discard the ones that contains less than 10 frames. 

For more information about **DHF1K**, click [here](https://github.com/wenguanwang/DHF1K "dhf1k"). See **DHF1K leaderboard** [here](https://mmcheng.net/videosal/ "dhf1k_lb")

### Download weights for testing or initialization of training
Google Drive: <https://drive.google.com/file/d/14EgtXJboEnrM19aL5i9gGPNKbus8790_/view?usp=sharing>

Baidu Disk: <https://pan.baidu.com/s/1jmRNufO_IXxJX4D0LKxTaQ>  password:pqil

The testing weights `U_td3_res_ds.h5` and `UHD_dcross_res_matt_res.h5` are put into '**\vap_model**' by default. The initialization weights `A_resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` and `M_resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` are put into '**\weights**' by default.

### Modify corresponding directories
Please modify the `config.py` accordingly.


##Testing

1.Easy testing: run `demo_test.py` to get result from one video by default. 
A video test example from **UCF sports** can be found in the **/DataSets** folder. The results would be stored in '**\vap_predictions**' by default.

2.To get other testing results, prepare the datasets with optical flows, and modify the dataset settings in `config.py`. Run `demo_test.py`. 
You may also run `demo.py` after editing the `config.py` with `model_no = 0 or 1` and `phase = 'vis'`， where `0` is for the feature net, and `1` is for the whole model.

3.To visualize the multi-scale masks or the side outputs, please modify the prediction part of `demo.py` or `demo_test.py`.



## Training

### Start training
Our model is trained in tendem. To train the feature net, we initialize it with the weight of resnet-50 pretrained on the ImageNet. Then, we initialize the feature net with the weight in the first step, randomly initialize the remaining part, and train the whole network.

In case that you do not want to train from the feature net, you may directly use the provided weight `U\_td3\_res_ds.h5`, and begin from step 2.

1.Train the feature net

1) In `config.py`, set `model_no = 0` and `phase = 'train'`

2) Run `demo.py`

2.Train the whole net

1) In `config.py`, set `model_no = 1` and `phase = 'train'`

2) Modify the initilization weight of the feature net in `demo.py` to the one you obtained in step 1, or leave it as the default one that you've downloaded.

3) Run `demo.py`


## Citation
If you use our code in your research work, please consider citing the following papers:

    @ARTICLE{lai2019video,
      title={Video Saliency Prediction using Spatiotemporal Residual Attentive Networks},
      author={Qiuxia Lai and Wenguan Wang and Hanqiu Sun and Jianbing Shen},
      journal={IEEE Trans. on Image Processing},
      year={2019}
    }

    @InProceedings{Wang_2018_CVPR,
    author = {Wang, Wenguan and Shen, Jianbing and Guo, Fang and Cheng, Ming-Ming and Borji, Ali},
    title = {Revisiting Video Saliency: A Large-Scale Benchmark and a New Model},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2018}
    }

    @ARTICLE{Wang_2019_revisitingVS, 
    author={W. {Wang} and J. {Shen} and J. {Xie} and M. {Cheng} and H. {Ling} and A. {Borji}}, 
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Revisiting Video Saliency Prediction in the Deep Learning Era}, 
    year={2019}, 
    }

##Contact
Qiuxia Lai: <qxlai@cse.cuhk.edu.hk> or <ashleylqx@gmail.com>

Wenguan Wang: <wenguanwang.ai@gmail.com>





