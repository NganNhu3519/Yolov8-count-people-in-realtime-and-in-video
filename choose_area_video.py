import cv2
import numpy as np

#Variable to save coordinates
areas = {"area1": [], "area2": []}
selected_area = "area1"  #Set up to draw area 1 first
resize_width = 1020  #Width 
resize_height = 500   #Height

#Callback to take the coordinates
def select_area(event, x, y, flags, param):
    global selected_area
    if event == cv2.EVENT_LBUTTONDOWN and len(areas[selected_area]) < 4:
        areas[selected_area].append((x, y))
        print(f"Added point {len(areas[selected_area])}: {x, y} for {selected_area}")
        if len(areas[selected_area]) == 4:
            print(f"{selected_area} is complete! Switch to the next area if needed.")

#Mouse
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', select_area)

cap = cv2.VideoCapture('1st.mp4')  #Put your name of your video

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))
    
    for area_name in areas:
        if len(areas[area_name]) == 4:
            cv2.polylines(frame, [np.array(areas[area_name], np.int32)], True, (255, 0, 0), 2)
            cv2.putText(frame, area_name[-1], tuple(areas[area_name][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    #Display screen
    cv2.imshow("Video", frame)

    #Press `n` to change the mode to draw area 2
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        if selected_area == "area1" and len(areas["area1"]) == 4:
            selected_area = "area2"
            print("Switching to area2...")
        else:
            print("Complete area1 first.")
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

#Coordinates will be displayed in the terminal
print("area1=", areas["area1"])
print("area2=", areas["area2"])