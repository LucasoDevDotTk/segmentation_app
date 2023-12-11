import warnings
# warnings.filterwarnings("ignore")

import mmcv
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import matplotlib.pyplot as plt

import torch

if torch.cuda.is_available():
    # Printing the number and name of cuda devices
    print("Number of cuda devices:", torch.cuda.device_count())
    print("Cuda device name:", torch.cuda.get_device_name(0))
    # Setting the default device to cuda
    device = torch.device("cuda")
else:
    raise "Torch GPU Error: No CUDA GPU Found"

def main():
    # Path to the config file
    config_file = './trained_models/rugd_group6/ganav_rugd_6.py'

    # Path to the checkpoint file
    checkpoint_file = './trained_models/rugd_group6/ganav_rugd_6.pth'

    # Initialize the model
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # Path to the image
    img = '/mnt/d/Programming/Git/private/ganav-test/GANav-offroad-test2/CWT/CWT/img/first/frame0000.jpg'

    # Run inference
    result = inference_segmentor(model, img)

    # Show/save results
    # Option 1: Display the result with matplotlib
    show_result_pyplot(model, img, result)  # Removed the get_palette part

    # Option 2: Save the result as an image file
    # model.show_result(img, result, out_file='result.jpg')

if __name__ == '__main__':
    main()