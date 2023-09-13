# import necessary libraries
from flask import Flask,render_template,Response
import cv2
import random #to create random color value
from ultralytics import YOLO
from deepsort import Tracker #deepSORT tracker class
import numpy as np


app = Flask(__name__)

# ========== OLD REPO CONTENTS ===========

# initiate YOLO type of models
model=YOLO('yolov8n-face.pt')

camera=cv2.VideoCapture('rtsp://admin:admin123@192.168.0.102/Streaming/Channels/402')

# opening object class list file
my_file = open("coco.txt", "r")
data = my_file.read()
# convert coco.txt file into python list
class_list = data.split("\n") 
#print(class_list) #check the list

# count variable initiation 
count=0
up_count = 0

people_list = {}
counted = set()

# initiate the tracker object from DeepSORT Tracker Class
tracker=Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# hardcoded line coordinate, will be changed soon
cy1=322
cy2=368
offset=6
# ========== OLD REPO CONTENTS ===========

# create frame generator for streaming the results
def gen_frames():
    while True:
        success,frame=camera.read()
        if not success:
            print("Error: Could not read frame from camera.")
            break
        else:
            frame=cv2.resize(frame,(640,480))
            # detect object using YOLO predictions
            results=model.predict(frame)
            # iterate each detections
            # could be an error from here because i was trying to use image frame to predict assuming it will be a frame by frame streaming
            for result in results:
                detections=[]
                for result_elements in result.boxes.data.tolist():
                    # converting each detections value into INT data type
                    # still dont know what x and y coordinates stands for
                    x1,y1,x2,y2,score,class_id=result_elements
                    x1=int(x1)
                    x2=int(x2)
                    y1=int(y1)
                    y2=int(y2)
                    class_id=int(class_id)
                    # append the (what is it when you extract the metadata) detection results into "detections" list
                    detections.append([x1, y1, x2, y2, score])
                # track the detected object
                tracker.update(frame,detections)
                # create a boundary box from the tracker for each detected frame
                for track in tracker.tracks:
                    bbox=track.bbox
                    x1,y1,x2,y2=bbox
                    # so it's a boundary box coordinates...
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    track_id=track.track_id

                    if y1>cy1:
                        # is it like append the identified object who cross the line?
                        people_list[track_id]=y1

                    # identify the tracked object 
                    cv2.putText(frame,(str(track_id)),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
                    # put the rectangle over the tracked object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
                    # display the list size from "people_list" representing how much people passing the count line
                    cv2.putText(frame, ("Count :"+str(len(people_list))),(500,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
                
                # create line 1
                cv2.line(frame,(0,cy1),(1280,cy1),(255,255,255),thickness=1)
                cv2.putText(frame, ("Line 1"),(280,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
                # create line 2
                cv2.line(frame,(0,cy2),(1280,cy2),(0,0,255),4)
                cv2.putText(frame, ("Line 2"),(100,cy2-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)


            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



# show this function when the route is on the landing page --> "/"
@app.route("/")
def index():
    # to render the html file i guess...
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')