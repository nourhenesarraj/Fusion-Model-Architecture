import cv2
import numpy as np
import glob
 
img_array = []
images = ["../data/ARCNNDataset/images/set01/V000/visible/frame" + str(i) + ".jpg" for i in range(401, 900)]

for filename in images:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
   
 
 
out = cv2.VideoWriter('/content/drive/MyDrive/robot-videos/vis_vid_3.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    
out.release()