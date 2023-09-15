# import necessary libraries
from flask import Flask,render_template,Response,request,jsonify,current_app
import cv2
import random #to create random color value
from ultralytics import YOLO
from deepsort import Tracker #deepSORT tracker class
import numpy as np
import mysql.connector
from datetime import datetime, time, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
app = Flask(__name__)

# ========== OLD REPO CONTENTS ===========

# initiate YOLO type of models
model=YOLO('yolov8n-face.pt')

camera=cv2.VideoCapture(0)

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



# ========== DATABASE RELATED FUNCTION ===========
# Initiate connection to MySQL server
mydb = mysql.connector.connect(
    host='localhost',
    user='root',
    password=''
)
# Instantiate cursor class for executing SQL commands
cursor = mydb.cursor()
database_name = "enpemo"
counter_table_name = "kehadiran"

# Will check if there's database named Enpemo exist on those server (server isn't it?)
def checkDatabaseExistance(database_name):
    check_db_query = f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{database_name}' " #sql query for checking if database named on 'database_name' variable is exist
    cursor.execute(check_db_query)
    result = cursor.fetchone()
    return result is not None

# Create a database 
def createDatabase(database_name):
    create_database_query = f"CREATE DATABASE {database_name}"#sql query for creating database named on 'database_name' variable
    cursor.execute(create_database_query)
    print(f"The database '{database_name}' has been created.")

# Function to execute the SQL Query which will be creating a new table if there's no table  
def createTableIfNotExist(table_name):
    # query for automatically checking and creating a table
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            `id` INT NOT NULL AUTO_INCREMENT,
            `jumlah` INT NOT NULL,
            `createdAt` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            `updatedAt` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 AUTO_INCREMENT=3;
    """
    cursor.execute(create_table_query)

def checkTableExistence(table_name):
    check_table_query = f"SHOW TABLES LIKE '{table_name}'" #sql query for searching table name
    cursor.execute(check_table_query) 
    result = cursor.fetchone() #fetch the search result -> 
    return result is not None #if the search result is not empty result, it'll returning not None value  

# # Function to insert data with button
# def dataInsertion():
#        global cursor, mydb  # Access the global cursor and mydb variables
#        # Get the counted people value
#        count = len(people_list)
#        # Current date to check if there's data already inserted
#        current_date = datetime.now().date()
#        # Check if a row already exists for the current date
#        check_query = f"SELECT * FROM kehadiran WHERE DATE(createdAt) = '{current_date}'"
#        cursor.execute(check_query)
#        existing_row = cursor.fetchone()
#        if existing_row:
#            # A row already exists for the current date, update it
#            update_query = f"UPDATE kehadiran SET jumlah = {count}, updatedAt = NOW() WHERE DATE(createdAt) = '{current_date}'"
#            cursor.execute(update_query)
#        else:
#            # No row exists for the current date, insert a new row
#            insert_query = f"INSERT INTO kehadiran (jumlah) VALUES ({count})"
#            cursor.execute(insert_query)
#        # Commit the changes to the database
#        mydb.commit()
#        # Return a JSON response indicating success
#        return jsonify({"message": "Data inserted or updated successfully"})


# Function to insert data with Time Trigger
def dataInsertion_TimeTrigger():
       global cursor, mydb  # Access the global cursor and mydb variables
       # Get the counted people value
       count = len(people_list)
       # Current date to check if there's data already inserted
       current_date = datetime.now().date()
       # Check if a row already exists for the current date
       check_query = f"SELECT * FROM kehadiran WHERE DATE(createdAt) = '{current_date}'"
       cursor.execute(check_query)
       existing_row = cursor.fetchone()
       if existing_row:
           # A row already exists for the current date, update it
           update_query = f"UPDATE kehadiran SET jumlah = {count}, updatedAt = NOW() WHERE DATE(createdAt) = '{current_date}'"
           cursor.execute(update_query)
       else:
           # No row exists for the current date, insert a new row
           insert_query = f"INSERT INTO kehadiran (jumlah) VALUES ({count})"
           cursor.execute(insert_query)
       # Commit the changes to the database
       mydb.commit()
       # Return a JSON response indicating success
       return True


# Global Logic -- Checking database existence then creating a database if there's no database found in the server
if not checkDatabaseExistance(database_name):
    createDatabase(database_name)
# Select the 'enpemo' database
cursor.execute(f"USE {database_name}")
createTableIfNotExist(counter_table_name)
# ========== DATABASE RELATED FUNCTION ===========

# ========== COUNTER RELATED FUNCTION ===========
# create frame generator for streaming the results
def generate_frame():
    # The Detection Segment
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
            # return len(people_list)
# ========== COUNTER RELATED FUNCTION ===========

# ========== TASK SCHEDULER RELATED FUNCTION (START) ===========
# Initialize the APScheduler
scheduler = BackgroundScheduler()

# Repeat your dataInsertion function every 2 seconds
scheduler.add_job(dataInsertion_TimeTrigger, 'interval', seconds=2)

# Start the scheduler when the Flask app starts
scheduler.start()

# Function to stop the scheduler
def cleanup():
    scheduler.shutdown()
    return jsonify({"message": "Data inserted or updated successfully"})
 

# ========== TASK SCHEDULER RELATED FUNCTION (END) ===========

# ========== MAIN FUNCTION and DB INITIATION===========


# show this function when the route is on the landing page --> "/"
@app.route("/")
def index():
    # to render the html file i guess...
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/insert_data',methods=['POST'])
# def insert_data():
#     if request.method=='POST':
#         result=dataInsertion()
#         return result    

# Add a stop-scheduling route
@app.route('/cleanup')
def cleanup_route():
    cleanup()
    return "Scheduler Stopped"

if __name__ == '__main__':
    app.run(debug=True)
