import cv2
import socket
import numpy as np

# Set up a socket client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.38', 3000))  # Replace 'MAC_IP_ADDRESS' with the actual IP of the Mac

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

    # Display the frame
    cv2.imshow('Remote Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
client_socket.close()
cv2.destroyAllWindows()
