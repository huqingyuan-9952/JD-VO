# Introduction

This repository contains a Python implementation of the JD-VO system (Joint Depth-Flow Visual Odometry), developed by Hu Qingyuan. 
It is based on the system described in **"Monocular Visual Odometry with Absolute Scale Recovery via Joint Depth-Flow Learning"**.


This repo includes
1. visual odometry system JD-VO;
2. evaluation scripts for visual odometry; 
3. visualize JD-VO results;
4. How to run JD-VO in your own dataset


### Contents
1. [Requirements](#part-1-requirements)
2. [Prepare dataset](#part-2-download-dataset-and-models)
3. [Run kitti odometry dataset](#part-3-run-kitti-odometry-dataset)
4. [Result evaluation](#part-4-result-evaluation)
5. [Run your own dataset](#part-5-run-your-own-dataset)


### Part 1. Requirements

This code was tested with Python 3.7, CUDA 11.6, Ubuntu 20.04, and [PyTorch-1.13](https://pytorch.org/).


```
conda create -n jd_vo python=3.7
conda activate jd_vo
```

### Part 2. Download dataset and models

The main dataset used in this project is [KITTI Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). After downloaing the dataset, our default settings expect that you have converted the png images to jpeg with this command, which also deletes the raw KITTI .png files:

```
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

and then resize to 640×192.


For models, please visit [here](https://www.dropbox.com/scl/fo/8xo2xsb3gv6o5ysm6c3gn/AMfeHtUQB6FeVjTvZozmdNQ?rlkey=mb8s9jxe64vphmr4tztdkjkp5&st=4jwuh3lp&dl=0) to download the models and save the models into the directory `models/`.

### Part 3. Run kitti odometry dataset

```
python apis/run.py -d options/examples/default_configuration.yml  
```

The result (trajectory pose file) is saved in `result_dir` defined in the configuration file.
Please check the `options/examples/default_configuration.yml` for reference. 

### Part 4. Result Evaluation

<div align="center">
  <img src='docs/00.png' width="80%">
  <p><b>Sequence 00</b></p>

  <img src='docs/02.png' width="80%">
  <p><b>Sequence 02</b></p>

  <img src='docs/05.png' width="80%">
  <p><b>Sequence 05</b></p>

  <img src='docs/07.png' width="80%">
  <p><b>Sequence 07</b></p>

  <img src='docs/09.png' width="80%">
  <p><b>Sequence 09</b></p>

  <img src='docs/10.png' width="80%">
  <p><b>Sequence 10</b></p>
</div>


#### KITTI
[KITTI Odometry benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) contains 22 stereo sequences, in which 11 sequences are provided with ground truth. The 11 sequences are used for evaluating visual odometry. The evaluation results can be obtained by running the following command:

```
python tools/evaluation/odometry/eval_odom.py --result result/tmp/0 --align 6dof
```
Alternatively, trajectory evaluation can also be performed using [evo](https://github.com/MichaelGrupp/evo).


### Part 5. Run your own dataset

To prepare your own dataset, the following steps are required:

1. Convert the images from PNG to JPEG format, and resize them to the target resolution.
2. Modify the camera intrinsic matrix accordingly to match the resized images. For convenience, you can directly replace the `calib.txt` files in the KITTI dataset with your updated calibration parameters.
3. Update the configuration `.yml` file to reflect the paths and parameters of the new dataset.
4. After completing the above steps, you can run the system with your own data.




### License
For academic usage, the code is released under the permissive MIT license. Our intension of sharing the project is for research/personal purpose. For any commercial purpose, please contact the authors. 


### Acknowledgement
Some of the codes were borrowed from the excellent works of [DF-VO](https://github.com/Huangying-Zhan/DF-VO),[monodepth2](https://github.com/nianticlabs/monodepth2), [LiteFlowNet](https://github.com/twhui/LiteFlowNet) and [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet). The borrowed files are licensed under their original license respectively.