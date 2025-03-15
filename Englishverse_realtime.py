import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *  # Import all code from the 'tracker' file
import threading  # Module for handling multi-threading
from datetime import datetime
import requests
import time  # Import library for using time.sleep()

# URL of the deployed Web App for data logging
url = 'https://script.google.com/macros/s/AKfycbw0_cWPCsr2hItu2KaLHKxwjqb1mudrWg4XH0UmkbGqg1vqmqTphYZd6_FYFgDxByJi/exec'

# Initialize the YOLO model (using a smaller version for faster processing)
model = YOLO('yolov8n.pt')  # Load the YOLOv8n model for detection

# Define areas of interest for counting (polygon coordinates)
area1 = [(345, 375), (406, 412), (495, 280), (440, 273)]  # Area 1: Entrance
area2 = [(385, 445), (464, 461), (582, 278), (510, 268)]  # Area 2: Exit

# Function to get coordinates when mouse moves over the frame
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check for mouse movement
        colorsBGR = [x, y]  # Get the x, y coordinates of the mouse
        print(colorsBGR)  # Print the coordinates

# Create a window for displaying the RGB frame
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)  # Set the mouse callback function

# Open a camera stream from a specified URL
camera_url = 'http://10.91.36.238:4747/video'
cap1 = cv2.VideoCapture(camera_url)  # Start capturing video from the camera

# Read the list of classes from the coco.txt file
my_file = open("coco.txt", "r")
data = my_file.read()  # Read the contents of the file
class_list = data.split("\n")  # Split the contents into a list of classes

# Initialize tracking variables
count = 0  # Frame count
tracker = Tracker()  # Initialize the tracker for object tracking
people_entering = {}  # Dictionary to track people entering
people_exiting = {}  # Dictionary to track people exiting
entering = set()  # Set to store IDs of people entering
exiting = set()  # Set to store IDs of people exiting

# Time interval for sending data to Google Sheets
last_send_time = time.time()  # Record the last send time

# Function to send data to Google Sheets
def send_to_sheet():
    global entering, exiting, last_send_time
    while True:  # Continuous loop to send data
        if time.time() - last_send_time > 5:  # Check if 5 seconds have passed
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),  # Current date
                'time': datetime.now().strftime('%H:%M:%S'),  # Current time
                'people_out': len(exiting),  # Number of people exiting
                'people_in': len(entering)   # Number of people entering
            }

            try:
                # Send a POST request to the Google Sheets API
                response = requests.post(url, json=data)

                # Check the response status
                if response.status_code == 200:
                    print(f"Data updated successfully: {data}")  # Success message
                else:
                    print(f"Error updating data. Status code: {response.status_code}")
                    print("Server response:", response.text)

            except Exception as e:
                print(f"Error sending request: {e}")  # Print any error encountered

            last_send_time = time.time()  # Update the last send time
        time.sleep(1)  # Reduce the frequency of requests

# Create a separate thread for sending data to Google Sheets
thread = threading.Thread(target=send_to_sheet)
thread.daemon = True  # Set the thread as a daemon
thread.start()  # Start the thread

# Function to reconnect to the camera if the connection is lost
def reconnect_camera():
    cap = cv2.VideoCapture(camera_url)  # Attempt to open the camera
    attempts = 0  # Initialize the number of attempts
    max_attempts = 5  # Maximum number of reconnection attempts
    while not cap.isOpened() and attempts < max_attempts:
        print("Unable to connect to the camera. Retrying...")
        attempts += 1  # Increment the attempt count
        time.sleep(5)  # Wait 5 seconds before retrying
        cap = cv2.VideoCapture(camera_url)  # Retry opening the camera
    if cap.isOpened():
        print("Reconnected to the camera successfully!")  # Success message
    else:
        print("Unable to reconnect to the camera after multiple attempts.")
    return cap  # Return the camera capture object

# Main loop for processing video and counting people entering/exiting
while True:
    ret, frame = cap1.read()  # Read a frame from the camera
    if not ret:
        print("Unable to get video from the source. Waiting to reconnect...")
        cap1 = reconnect_camera()  # Call the reconnect function if the connection is lost
        continue  # If no frame is received, continue the loop

    count += 1  # Increment the frame count
    if count % 3 != 0:
        continue  # Reduce processing load, only process every 3rd frame

    # Resize the video frame for faster processing while maintaining quality
    frame = cv2.resize(frame, (680, 520))  # Set new dimensions for the video frame

    # Make predictions using the YOLO model
    results = model.predict(frame)

    a = results[0].boxes.data  # Extract bounding box data from the results
    px = pd.DataFrame(a).astype("float")  # Convert the bounding box data to a DataFrame
    list = []  # Initialize a list for bounding boxes

    # Filter objects that are classified as 'person'
    for index, row in px.iterrows():
        x1 = int(row[0])  # Get the x-coordinate of the top-left corner
        y1 = int(row[1])  # Get the y-coordinate of the top-left corner
        x2 = int(row[2])  # Get the x-coordinate of the bottom-right corner
        y2 = int(row[3])  # Get the y-coordinate of the bottom-right corner
        d = int(row[5])  # Get the class ID
        c = class_list[d]  # Get the class name
        if 'person' in c:  # Check if the detected object is a person
            list.append([x1, y1, x2, y2])  # Add the bounding box to the list
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the detected person
            cv2.putText(frame, str('person'), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)  # Label the bounding box

    bbox_id = tracker.update(list)  # Update the tracker with the current bounding boxes
    
    # Process entering and exiting people
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox  # Unpack the bounding box coordinates and ID

        #### Handle entering people ###
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)  # Check if the bottom-right corner is in area 2
        if results >= 0:  # If the point is inside the area
            people_entering[id] = (x4, y4)  # Track the person as entering
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Draw a rectangle around the person

        if id in people_entering:  # If the person is already tracked as entering
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)  # Check if the person is now in area 1
            if results1 >= 0:  # If the point is inside area 1
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw a rectangle in green
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)  # Draw a circle at the bottom-right corner
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)  # Label the ID
                entering.add(id)  # Add the ID to the entering set

        #### Handle exiting people ###
        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)  # Check if the person is in area 1
        if results2 >= 0:  # If the point is inside area 1
            people_exiting[id] = (x4, y4)  # Track the person as exiting
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw a rectangle in green

        if id in people_exiting:  # If the person is already tracked as exiting
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)  # Check if the person is now in area 2
            if results3 >= 0:  # If the point is inside area 2
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)  # Draw a rectangle in magenta
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)  # Draw a circle at the bottom-right corner
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)  # Label the ID
                exiting.add(id)  # Add the ID to the exiting set

    # Draw the areas on the frame
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)  # Draw area 1 outline
    cv2.putText(frame, str('1'), (459, 366), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)  # Label area 1
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)  # Draw area 2 outline
    cv2.putText(frame, str('2'), (399, 387), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)  # Label area 2

    # Display the number of people entering and exiting
    i = len(entering)  # Count the number of people entering
    o = len(exiting)  # Count the number of people exiting
    cv2.putText(frame, "Enter: " + str(i), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)  # Show entering count
    cv2.putText(frame, "Exit: " + str(o), (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)  # Show exiting count

    cv2.imshow("RGB", frame)  # Display the frame with all annotations

    # Exit the program if the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources when exiting
cap1.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows