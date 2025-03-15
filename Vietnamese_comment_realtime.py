import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *  # Nhập tất cả mã từ tệp 'tracker'
import threading  # Thư viện để xử lý đa luồng
from datetime import datetime
import requests
import time  # Nhập thư viện để sử dụng hàm time.sleep()

# URL của Web App đã triển khai để ghi dữ liệu
url = 'https://script.google.com/macros/s/AKfycbw0_cWPCsr2hItu2KaLHKxwjqb1mudrWg4XH0UmkbGqg1vqmqTphYZd6_FYFgDxByJi/exec'

# Khởi tạo mô hình YOLO (sử dụng phiên bản nhỏ hơn để xử lý nhanh hơn)
model = YOLO('yolov8n.pt')  # Tải mô hình YOLOv8n để phát hiện

# Định nghĩa các khu vực quan tâm cho việc đếm (tọa độ đa giác)
area1 = [(345, 375), (406, 412), (495, 280), (440, 273)]  # Khu vực 1: Cổng vào
area2 = [(385, 445), (464, 461), (582, 278), (510, 268)]  # Khu vực 2: Cổng ra

# Hàm lấy tọa độ khi chuột di chuyển trên khung hình
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Kiểm tra sự kiện di chuyển chuột
        colorsBGR = [x, y]  # Lấy tọa độ x, y của chuột
        print(colorsBGR)  # In ra tọa độ

# Tạo cửa sổ để hiển thị khung hình RGB
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)  # Thiết lập hàm callback cho chuột

# Mở luồng camera từ URL đã chỉ định
camera_url = 'http://10.91.36.238:4747/video'
cap1 = cv2.VideoCapture(camera_url)  # Bắt đầu ghi hình từ camera

# Đọc danh sách các lớp từ tệp coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()  # Đọc nội dung của tệp
class_list = data.split("\n")  # Chia nội dung thành danh sách các lớp

# Khởi tạo các giá trị theo dõi
count = 0  # Đếm số khung hình
tracker = Tracker()  # Khởi tạo bộ theo dõi cho việc theo dõi đối tượng
people_entering = {}  # Từ điển theo dõi người vào
people_exiting = {}  # Từ điển theo dõi người ra
entering = set()  # Tập hợp lưu ID của người vào
exiting = set()  # Tập hợp lưu ID của người ra

# Thời gian gửi dữ liệu lên Google Sheets
last_send_time = time.time()  # Ghi lại thời gian gửi lần cuối

# Hàm gửi dữ liệu lên Google Sheets
def send_to_sheet():
    global entering, exiting, last_send_time
    while True:  # Vòng lặp liên tục để gửi dữ liệu
        if time.time() - last_send_time > 5:  # Kiểm tra xem đã 5 giây chưa
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),  # Ngày hiện tại
                'time': datetime.now().strftime('%H:%M:%S'),  # Thời gian hiện tại
                'people_out': len(exiting),  # Số người đang ra ngoài
                'people_in': len(entering)   # Số người đang vào trong
            }

            try:
                # Gửi yêu cầu POST tới API Google Sheets
                response = requests.post(url, json=data)

                # Kiểm tra trạng thái phản hồi
                if response.status_code == 200:
                    print(f"Dữ liệu đã được cập nhật thành công: {data}")  # Thông báo thành công
                else:
                    print(f"Có lỗi khi cập nhật dữ liệu. Mã lỗi: {response.status_code}")
                    print("Phản hồi từ máy chủ:", response.text)

            except Exception as e:
                print(f"Lỗi khi gửi yêu cầu: {e}")  # In ra bất kỳ lỗi nào gặp phải

            last_send_time = time.time()  # Cập nhật thời gian gửi dữ liệu lần cuối
        time.sleep(1)  # Giảm tần suất gửi yêu cầu

# Tạo một luồng riêng để gửi dữ liệu lên Google Sheets
thread = threading.Thread(target=send_to_sheet)
thread.daemon = True  # Đặt luồng là daemon
thread.start()  # Bắt đầu luồng

# Hàm kiểm tra và kết nối lại camera nếu mất kết nối
def reconnect_camera():
    cap = cv2.VideoCapture(camera_url)  # Thử mở camera
    attempts = 0  # Khởi tạo số lần thử
    max_attempts = 5  # Số lần thử tối đa
    while not cap.isOpened() and attempts < max_attempts:
        print("Không thể kết nối tới camera. Đang thử lại...")
        attempts += 1  # Tăng số lần thử
        time.sleep(5)  # Đợi 5 giây trước khi thử lại
        cap = cv2.VideoCapture(camera_url)  # Thử mở camera lại
    if cap.isOpened():
        print("Kết nối lại với camera thành công!")  # Thông báo thành công
    else:
        print("Không thể kết nối lại với camera sau nhiều lần thử.")
    return cap  # Trả về đối tượng ghi hình camera

# Vòng lặp chính để xử lý video và đếm người vào/ra
while True:
    ret, frame = cap1.read()  # Đọc một khung hình từ camera
    if not ret:
        print("Không thể lấy video từ nguồn. Đang chờ kết nối lại...")
        cap1 = reconnect_camera()  # Gọi hàm kết nối lại nếu mất kết nối
        continue  # Nếu không có khung hình, tiếp tục vòng lặp

    count += 1  # Tăng số khung hình
    if count % 3 != 0:
        continue  # Giảm tải xử lý, chỉ xử lý mỗi 3 khung hình một lần

    # Giảm độ phân giải video để xử lý nhanh hơn mà vẫn giữ chất lượng
    frame = cv2.resize(frame, (680, 520))  # Đặt kích thước mới cho khung hình video

    # Dự đoán với mô hình YOLO
    results = model.predict(frame)

    a = results[0].boxes.data  # Lấy dữ liệu bounding box từ kết quả
    px = pd.DataFrame(a).astype("float")  # Chuyển đổi dữ liệu bounding box thành DataFrame
    list = []  # Khởi tạo danh sách cho các bounding box

    # Lọc các đối tượng là 'person' (người)
    for index, row in px.iterrows():
        x1 = int(row[0])  # Lấy tọa độ x của góc trên bên trái
        y1 = int(row[1])  # Lấy tọa độ y của góc trên bên trái
        x2 = int(row[2])  # Lấy tọa độ x của góc dưới bên phải
        y2 = int(row[3])  # Lấy tọa độ y của góc dưới bên phải
        d = int(row[5])  # Lấy ID lớp
        c = class_list[d]  # Lấy tên lớp
        if 'person' in c:  # Kiểm tra xem đối tượng phát hiện có phải là người không
            list.append([x1, y1, x2, y2])  # Thêm bounding box vào danh sách
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật quanh người phát hiện
            cv2.putText(frame, str('person'), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)  # Ghi nhãn cho bounding box

    bbox_id = tracker.update(list)  # Cập nhật bộ theo dõi với các bounding box hiện tại
    
    # Xử lý người vào và người ra
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox  # Giải nén tọa độ bounding box và ID

        #### Xử lý người vào ###
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)  # Kiểm tra xem góc dưới bên phải có nằm trong khu vực 2 không
        if results >= 0:  # Nếu điểm nằm trong khu vực
            people_entering[id] = (x4, y4)  # Theo dõi người vào
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Vẽ hình chữ nhật quanh người

        if id in people_entering:  # Nếu người đã được theo dõi là người vào
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)  # Kiểm tra xem người có ở khu vực 1 không
            if results1 >= 0:  # Nếu điểm nằm trong khu vực 1
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Vẽ hình chữ nhật màu xanh
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)  # Vẽ hình tròn tại góc dưới bên phải
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)  # Ghi nhãn ID
                entering.add(id)  # Thêm ID vào tập hợp người vào

        #### Xử lý người ra ###
        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)  # Kiểm tra xem người có ở khu vực 1 không
        if results2 >= 0:  # Nếu điểm nằm trong khu vực 1
            people_exiting[id] = (x4, y4)  # Theo dõi người ra
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Vẽ hình chữ nhật màu xanh

        if id in people_exiting:  # Nếu người đã được theo dõi là người ra
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)  # Kiểm tra xem người có ở khu vực 2 không
            if results3 >= 0:  # Nếu điểm nằm trong khu vực 2
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)  # Vẽ hình chữ nhật màu magenta
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)  # Vẽ hình tròn tại góc dưới bên phải
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)  # Ghi nhãn ID
                exiting.add(id)  # Thêm ID vào tập hợp người ra

    # Vẽ các khu vực trên khung hình
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)  # Vẽ đường viền khu vực 1
    cv2.putText(frame, str('1'), (459, 366), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)  # Ghi nhãn khu vực 1
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)  # Vẽ đường viền khu vực 2
    cv2.putText(frame, str('2'), (399, 387), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)  # Ghi nhãn khu vực 2

    # Hiển thị số người vào và ra
    i = len(entering)  # Đếm số người đang vào
    o = len(exiting)  # Đếm số người đang ra
    cv2.putText(frame, "Enter: " + str(i), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)  # Hiển thị số người vào
    cv2.putText(frame, "Exit: " + str(o), (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)  # Hiển thị số người ra

    cv2.imshow("RGB", frame)  # Hiển thị khung hình với tất cả các chú thích

    # Thoát khỏi chương trình nếu nhấn phím ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng tài nguyên khi thoát
cap1.release()  # Giải phóng đối tượng ghi hình camera
cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV