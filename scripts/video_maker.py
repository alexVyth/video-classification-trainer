import os
import cv2 as cv
import numpy as np
import glob




CATEGORY_DICT = {
    '0': 'Climbing',
    '2': 'Immobility',
    '1': 'Swimming',
    '3': 'Diving',
    '5': 'Not Scored',
    '4':'Head Shake'
}

video_directory = 'video.AVI'
#ftiaxnw video me neo megethos kai to apothikeuw edw
'''
img_array = []
for filename in glob.glob('./input_video/4/*.jpg'):
    img = cv.imread(filename)
    
    dim = (300,400)
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    height, width, layers = resized.shape
    size = (width,height)
    img_array.append(resized)


out = cv.VideoWriter('video.avi',cv.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
'''

correct_pred_collor = (0,255,0)
wrong_pred_collor = (0,0,255)
sample_count = 0
frame_count = 0
text_list = []


with open('results.txt', mode="r", encoding="utf-8") as file:
    for line in file:
        #pred, true = line.split()
        txt = line.split()
        text_list.append(txt)
        print(text_list)






cap = cv.VideoCapture(video_directory)
out = cv.VideoWriter('final_video.avi',cv.VideoWriter_fourcc(*'DIVX'), 25, (300,400))

if (cap.isOpened()== False): 
  print("Error opening video stream or file")


while(cap.isOpened()):
  
  ret, frame = cap.read()
  if ret == True:
    
    frame_count += 1
    sample_count = frame_count//32
    pred, true = text_list[sample_count]
    print(CATEGORY_DICT[pred])
   
    if pred==true:
        cv.putText(img=frame,text=CATEGORY_DICT[pred], org=(0, 30),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,0,255),thickness=2,lineType=cv.LINE_AA)
        cv.putText(img=frame,text=CATEGORY_DICT[true], org=(150, 30),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,255,0),thickness=2,lineType=cv.LINE_AA)
        cv.rectangle(frame,(50,50),(200,300),correct_pred_collor,1)
    else:
        
        cv.putText(img=frame,text=CATEGORY_DICT[pred], org=(0, 30),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,0,255),thickness=2,lineType=cv.LINE_AA)
        cv.putText(img=frame,text=CATEGORY_DICT[true], org=(150, 30),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,255,0),thickness=2,lineType=cv.LINE_AA)
        cv.rectangle(frame,(50,50),(200,300),wrong_pred_collor,1)
   

    
    
    cv.imshow('Frame',frame)
    out.write(frame)
    
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  
  else: 
    break


out.release()
cap.release()


cv.destroyAllWindows()
        



