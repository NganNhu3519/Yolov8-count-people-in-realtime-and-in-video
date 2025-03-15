import cv2
import numpy as np

#Variable to save coordinates
areas = {"area1": [], "area2": []}
selected_area = "area1"  #Set up to draw area 1 first

#Callback to take the coordinates
def select_area(event, x, y, flags, param):
    global selected_area
    if event == cv2.EVENT_LBUTTONDOWN and len(areas[selected_area]) < 4:
        areas[selected_area].append((x, y))
        print(f"Added point {len(areas[selected_area])}: {x, y} for {selected_area}")
        if len(areas[selected_area]) == 4:
            draw_rectangle(selected_area)

#Function draw in real time
def draw_rectangle(area_name):
    if area_name in areas and len(areas[area_name]) == 4:
        pts = np.array(areas[area_name], np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
        cv2.putText(frame, area_name[-1], tuple(pts[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

#Mouse
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_area)

#Read image
image_path = 'pan.jpg'  #Put your name of your image
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image. Check the file path!")
    exit()

frame = cv2.resize(frame, (1020, 500))  #Resize frame if you want

while True:
    #Display the image
    display_frame = frame.copy()  #Create a copy to not change the original image
    for area_name in areas:
        if len(areas[area_name]) > 0:
            pts = np.array(areas[area_name], np.int32)
            cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
            if len(areas[area_name]) == 4:
                cv2.putText(display_frame, area_name[-1], tuple(pts[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Image", display_frame)

    #Press `n` to change the mode to draw area 2
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        if selected_area == "area1" and len(areas["area1"]) == 4:
            selected_area = "area2"
            print("Switching to area2...")
        else:
            print("Complete area1 first.")
    elif key == 27:  #Press ESC to exit
        break

cv2.destroyAllWindows()

#Coordinates will be displayed in the terminal
print("area1=", areas["area1"])
print("area2=", areas["area2"])