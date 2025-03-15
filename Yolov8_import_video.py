import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *  # Import all code from file 'tracker'
import threading  # Module to process multiple threads
from datetime import datetime
import requests
import time 

# URL of your Google Sheets
url = 'https://script.google.com/macros/s/AKfycbxhdTYhn3hFGqTNTgGlKJ7fEdi_h6BONWZmsMMAfBtKz6HOV1c-vQGtifjGqF9JPBfW/exec'

#model yolov8s is smaller than yolov8 to optimize for your GPU
model=YOLO('yolov8s.pt')

#coordinates of area, x1,,x2,x3,...
area1= [(491, 260), (451, 274), (612, 347), (620, 310)]
area2= [(426, 282), (352, 298), (567, 487), (659, 381)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#import video, you should put name of video in ()
cap=cv2.VideoCapture('1st.mp4')

#coco file is a file of label
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

#Define value of counting
count=0
tracker = Tracker()
people_entering = {}
people_exiting = {}
entering=set()
exiting=set()

#Time to send data to Google Sheets
last_send_time = time.time()

# Function to send data to Google Sheets
def send_to_sheet():
    global entering, exiting, last_send_time
    while True:
        if time.time() - last_send_time > 5:  #Send data every 5s
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S'),
                'people_in': len(entering),   #Number people enter
                'people_out': len(exiting)  #Number people exit
            }

            try:
                #Send request POST
                response = requests.post(url, json=data)

                #Check the result
                if response.status_code == 200:
                    print(f"Dữ liệu đã được cập nhật thành công: {data}")
                else:
                    print(f"Có lỗi khi cập nhật dữ liệu. Mã lỗi: {response.status_code}")
                    print("Phản hồi từ server:", response.text)

            except Exception as e:
                print(f"Lỗi khi gửi yêu cầu: {e}")

            last_send_time = time.time()  #Update the last time updating data
        time.sleep(1)  #Reduce number of each request

#Create a seperate thread to send data to Google Sheets
thread = threading.Thread(target=send_to_sheet)
thread.daemon = True
thread.start()

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
           list.append([x1,y1,x2,y2])
           cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.circle(frame,(x2,y2),4,(255,0,255),-1)
           cv2.putText(frame,str('person'),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
    bbox_id=tracker.update(list)
    
for bbox in bbox_id:
    x3, y3, x4, y4, id = bbox

    #Cound the central coordinates of bounding box
    center_x = int((x3 + x4) / 2)
    center_y = int((y3 + y4) / 2)

    #Check people entering (area2 -> area1)
    results = cv2.pointPolygonTest(np.array(area2, np.int32), (center_x, center_y), False)
    if results >= 0:
        people_entering[id] = (center_x, center_y)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Vẽ tâm bounding box

    if id in people_entering:
        results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (center_x, center_y), False)
        if results1 >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Vẽ tâm bounding box
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            entering.add(id)

    #Check people exiting(area1 -> area2)
    results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (center_x, center_y), False)
    if results2 >= 0:
        people_exiting[id] = (center_x, center_y)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Vẽ tâm bounding box

    if id in people_exiting:
        results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (center_x, center_y), False)
        if results3 >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)  # Vẽ tâm bounding box
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            exiting.add(id)
    
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(459, 366),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(399, 387),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
    
    #print(people_entering)
    i = (len(entering))
    o = (len(exiting))
    cv2.putText(frame,"Enter: " + str(i),(60, 80),cv2.FONT_HERSHEY_COMPLEX,(0.7),(0,0,255),2)
    cv2.putText(frame,"Exit: " + str(o),(60, 140),cv2.FONT_HERSHEY_COMPLEX,(0.7),(255,0,255),2)
    
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
