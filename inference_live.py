import cv2
import socket
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor

import torch

if torch.cuda.is_available():
    # Printing the number and name of cuda devices
    print("Number of cuda devices:", torch.cuda.device_count())
    print("Cuda device name:", torch.cuda.get_device_name(0))
    # Setting the default device to cuda
    device = torch.device("cuda:0")
else:
    # Printing a message that cuda is not available
    print("Cuda is not available")
    exit()


# Set up a socket client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.38', 3000))  # Replace 'MAC_IP_ADDRESS' with the actual IP of the Mac

# Path to the config file and checkpoint file
config_file = './trained_models/rugd_group6/ganav_rugd_6.py'
checkpoint_file = './trained_models/rugd_group6/ganav_rugd_6.pth'

# Initialize the model
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

while True:
    # Receive the size of the frame
    size_data = client_socket.recv(4)
    if not size_data:
        break

    # Convert the size data to an integer
    frame_size = int.from_bytes(size_data, byteorder='big')

    # Receive the frame data
    frame_data = b""
    while len(frame_data) < frame_size:
        packet = client_socket.recv(min(frame_size - len(frame_data), 4096))
        if not packet:
            break
        frame_data += packet

    # Break the loop if no more data is received
    if not frame_data:
        break

    # Deserialize the frame using cv2.imdecode
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), 1)

    # Run inference
    result = inference_segmentor(model, frame)

    # Convert the segmentation result to a BGR image for visualization
    result_img = model.show_result(frame, result, out_file=None, show=False)

    # Display the original frame and the segmentation result side by side
    cv2.imshow('Remote Video', np.hstack([frame, result_img]))

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
client_socket.close()
cv2.destroyAllWindows()
