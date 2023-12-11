# Segmentation App [ALPHA]

## Setup

1. Install all dependecies
###### Step 1.1: Create Conda Environment

```
conda create -n ganav python=3.7 -y
conda activate ganav
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
# or use 
# conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
```

###### Step 1.2: Installing MMCV (1.3.16)

```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# or use
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
Note: Make sure you mmcv version is compatible with your pytorch and cuda version. In addition, you can specify the MMCV verion (1.3.16).

###### Step 1.3: Installing GANav
```
git clone https://github.com/rayguan97/GANav-offroad.git
cd GANav-offroad
pip install einops prettytable
pip install -e . 
```
2. Place .pth file in `./trained_models/rugd_group6/ganav_rugd_6.pth`
3. Change IP in `camera_server.py` to reflect the IP of the physical server.
4. Run `camera_server.py` on the machine with the camera
5. Change IP in `inference_live.py` to reflect the IP of the physical server. (The IP of the machine you ran `camera_server.py` on)
6. Run `inference_live.py`

## Troubleshooting
1. Check firewall on both machines
2. Both machine must be on same network

## Credits
We're using [GANav-offroad](https://github.com/rayguan97/GANav-offroad) which is licensed under the Apache 2.0 license. A copy of it can be found `credits/GANav-offroad`

[GANav-offroad](https://github.com/rayguan97/GANav-offroad) is heavely based on MMSegmentation, a copy of that license can be found in `credits/MMSegmentation`

## License
This project is licensed under the MIT License.