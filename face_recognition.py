import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
path = "known/"
imgList = os.listdir(path)
total = len(imgList)

path2 = "check/"
imgList2 = os.listdir(path2)
total2 = len(imgList2)
# imgTest = fr.load_image_file(pathTest)   
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
# faceLocTest = fr.face_locations(imgTest)[0]
# encodeTest = fr.face_encodings(imgTest)[0]
# result = fr.compare_faces([encodeElon], encodeTest)

def crop(img):
    img = cv2.resize(img, (640, 480))
    return img

classNames = []
images = []
def getEncodings():
    faces = []
    for cl in imgList:
        curImg = cv2.imread(f"{path}{cl}")
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        faces.append(encode)

    return faces

def markAttendance(name):
    with open(path + "attendance.csv" , "r+") as f:
        myDataList = f.readLines()
        nameList = []
        for line in myDataList:
            entry = list.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name}{dtString}")

# markAttendance("Elon")
def check(faces):
    for i in range(total2):
        imgr = fr.load_image_file(path2 + imgList[i])
        imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
        # cv2.imshow(imgr) 
        encodeTest = fr.face_encodings(imgr)[0]
        result = fr.compare_faces(faces, encodeTest)
        faceDis = fr.face_distance(faces, encodeTest)
        for a in faceDis:
            print(a)
            if a <= 0.6: cv2.imshow(imgr), print("EXIST") 
            else: print(i+1), print("Does NOt Exist")
        # if result[0] == True:
        #     cv2.imshow(imgr)
        # elif faceDis[0] < 0.7: cv2.imshow(imgr) 

def fromCam(encodeList):
    cap = cv2.VideoCapture(0)
    while 1:
        flag, img = cap.read()
        #   Scaling down to 0.25 wt and Ht
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0,25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = fr.face_locations(imgS)
        encodeFrame = fr.face_encodings(imgS, faceCurFrame)

        for encodeFace,faceLoc in zip(encodeFrame, faceCurFrame):
            matches = fr.compare_faces(encodeList, encodeFace)
            faceDis = fr.face_distance(encodeList, encodeFace)
            # Return indice of minValue in array
            matchInd = np.argmin(faceDis)
            if matches[matchInd]:
                name = classNames[matchInd].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                
                markAttendance(name)

# print(imgList)
encodeList = getEncodings()
# print("Done")
# print(classNames)
# print(len(encodeList))
# 
# Check if same person pic exist in path2 
check(encodeList)

# Mark attendance From Camera
fromCam(encodeList)
    



