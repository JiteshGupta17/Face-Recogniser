import cv2
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

file_name=input("enter the name")
face_data=[]

dataset_path='C:\\Users\\jites\\Desktop\\face_recognition1\\data\\'
skip=0

while True:  
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    #cv2.imshow("Video Frame",frame)
    faces=sorted(faces,key=lambda f:f[2]*f[3])

    #pick the largest face
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        #cv2.imshow("Frame",frame)
        #section out the region of interest
        offset=10
        frame_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        frame_section=cv2.resize(frame_section,(100,100))
        skip+=1
        if skip%10==0:
            face_data.append(frame_section)
            print(len(face_data))

    cv2.imshow("Video Frame",frame)
    cv2.imshow("gray Frame",gray_frame)  

    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))

print(face_data.shape)
np.save(dataset_path +file_name +'.npy',face_data)
print("data saved sucessfully at :",dataset_path +file_name +'.npy')
    
cap.release()
cv2.destroyAllWindows() 