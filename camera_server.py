import cv2
import socket

# Open a video capture object (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up a socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.1.38', 3000))  # Use 0.0.0.0 to bind to all available interfaces
server_socket.listen(2)
print("Server listening on port 3000...")

# Accept a connection
client_socket, client_address = server_socket.accept()
print("Connection from:", client_address)

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Serialize the frame using cv2.imencode
    _, frame_data = cv2.imencode('.jpg', frame)

    # Send the size of the frame first
    client_socket.sendall(len(frame_data).to_bytes(4, byteorder='big'))

    # Send the frame data
    client_socket.sendall(frame_data.tobytes())

# Clean up
client_socket.close()
server_socket.close()
cap.release()