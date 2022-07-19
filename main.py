import cv2

#OpenCV DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale = 1/255)

#Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

cap = cv2.VideoCapture(0) #selects capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#FULL HD
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()  #Reads frame from selected campture device

    #Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
       (x,y,w,h) = bbox
       class_name = classes[class_id]

       cv2.putText(frame, class_name, (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (200,0,50), 2)
       cv2.rectangle(frame, (x,y), (x+w,y+h),(200,0,50), 3)

    cv2.imshow("Frame", frame) #Shows frame with Window title

    cv2.waitKey(1) #Holds window from closing

