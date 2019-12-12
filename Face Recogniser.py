import cv2
import numpy as np
import os

def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())
def knn(training_set,test,k=5):
    ix=trainset[:,:-1]
    iy=trainset[: ,-1]
    m=trainset.shape[0]
    vals=[]
    for i in range(m):
        d=distance(ix[i],test)
        vals.append([d,iy[i]])
    vals=sorted(vals)
    vals=vals[:k]
    new_vals=np.unique(vals,return_counts=True)
    #print(new_vals.shape)
    index=new_vals[1].argmax()
    predict=new_vals[0][index]
    return predict
#initiate camera
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
class_id=0
name={}
face_data=[]
labels=[]
dataset_path='C:\\Users\\jites\\Desktop\\face_recognition1\\data\\'
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #mapping of name and class id
        name[class_id]=fx[ :-4]
        data_item=np.load(dataset_path + fx)
        face_data.append(data_item)
        #labels
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
        
face_dataset=np.concatenate(face_data,axis=0)
label_dataset=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(label_dataset.shape)
trainset=np.concatenate((face_dataset,label_dataset),axis=1)
print(trainset.shape)
while True:
    ret,frame=cap.read()
    #gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    #pick the largest face
    for face in faces:
        x,y,w,h=face
        
        #cv2.imshow("Frame",frame)
        #section out the region of interest
        offset=10
        frame_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        frame_section=cv2.resize(frame_section,(100,100))
        out=knn(trainset,frame_section.flatten(),k=5)
        pred_name=name[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(250,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    
    cv2.imshow("Video Frame",frame)
    #cv2.imshow("gray Frame",gray_frame)  
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()