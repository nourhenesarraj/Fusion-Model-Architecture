import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from google.colab.patches import cv2_imshow
# from vidgear.gears import WriteGear , VideoGear
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont

import config
from datasets import KAISTPed

import torchvision.transforms as T
from torchvision.transforms import Grayscale
from utils.transforms import FusionDeadZone, Compose, Resize, ToTensor
from torchvision.utils import save_image



# Label map
voc_labels = ('P', 'M', 'A', 'B', 'a')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#3cb44b', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def _line(Draw, xy, dot=0, width=3, fill='#3cb44b', h=False):
    if dot != 0:
        if (xy[2] - xy[0]) > (xy[3] - xy[1]):
            h=True
        if h:
            for i in range(xy[0], xy[2]):
                if i%dot*2==0:
                    Draw.line((i, xy[1], i+4, xy[3]), width=width, fill=fill)
        else:
            for i in range(xy[1], xy[3]):
                if i%dot*2==0:
                    Draw.line((xy[0], i, xy[2], i+4), width=width, fill=fill)
    else:
        Draw.line(xy, width=width, fill=fill)

def rectangle(draw, rec, dot=0, width=2, fill='#3cb44b' ):
    rec = np.array(rec, dtype=np.int16)
    a = tuple(rec[0:2])
    b = tuple([rec[2], rec[1]])
    c = tuple([rec[0], rec[3]])
    d = tuple(rec[2:4])

    _line(draw, a+b, fill=fill, dot=dot, width=width)
    _line(draw, a+c, fill=fill, dot=dot, width=width)
    _line(draw, c+d, fill=fill, dot=dot, width=width)
    _line(draw, b+d, fill=fill, dot=dot, width=width)

def detect(original_image, original_lwir, detection, \
        min_score, max_overlap=0.425, top_k=200, \
        suppress=None, width=2):
      """
      Detect objects in an image with a trained SSD300, and visualize the results.

      :param original_image: image, a PIL Image
      :param min_score: minimum threshold for a detected box to be considered a match for a certain class
      :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
      :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
      :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
      :return: annotated image, a PIL Image
      """

      det_boxes = detection[:,1:5]
      small_object =  det_boxes[:, 3] < 55
      det_boxes[:,2] = det_boxes[:,0] + det_boxes[:,2]

      det_boxes[:,3] = det_boxes[:,1] + det_boxes[:,3] 
     
      det_scores = detection[:,5]

      det_labels = list()
      for i in range(len(detection)) : 
          det_labels.append(1.0)
      det_labels = np.array(det_labels)
      det_score_sup = det_scores < min_score   
      det_boxes = det_boxes[~det_score_sup]
      det_scores = det_scores[~det_score_sup]
      det_labels = det_labels[~det_score_sup]
      
      # Decode class integer labels
      det_labels = [rev_label_map[l] for l in det_labels]

      # PIL from Tensor
      original_image = original_image.squeeze().permute(1, 2, 0)
      original_image = original_image.numpy() * 255
      original_lwir = original_lwir.squeeze().numpy() * 255
      original_image = Image.fromarray(original_image.astype(np.uint8))
      original_lwir = Image.fromarray(original_lwir.astype(np.uint8))
      
      # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
      if det_labels == ['background']:
          # Just return original image
          new_image = Image.new('RGB',(2*original_image.size[0], original_image.size[1]))
          new_image.paste(original_image,(0,0))
          new_image.paste(original_lwir,(original_image.size[0],0))
          return new_image

      # Annotate
      annotated_image = original_image
      annotated_image_lwir = original_lwir
      draw = ImageDraw.Draw(annotated_image)
      draw_lwir = ImageDraw.Draw(annotated_image_lwir)
      font = ImageFont.truetype("./utils/calibril.ttf", 15)

      # Suppress specific classes, if needed
      for i in range(det_boxes.shape[0]):
          if suppress is not None:
              if det_labels[i] in suppress:
                  continue
                  
          # Boxes
          box_location = det_boxes[i].tolist()
          draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]], width=width)
          draw_lwir.rectangle(xy=box_location, outline=label_color_map[det_labels[i]], width=width)
          
          # Text       
          text_score_vis = str(det_scores[i].item())[:7]
          text_score_lwir = str(det_scores[i].item())[:7]
          
          text_size_vis = font.getsize(text_score_vis)
          text_size_lwir = font.getsize(text_score_lwir)

          text_location_vis = [box_location[0] + 2., box_location[1] - text_size_vis[1]]
          textbox_location_vis = [box_location[0], box_location[1] - text_size_vis[1], box_location[0] + text_size_vis[0] + 4.,box_location[1]]
          
          text_location_lwir = [box_location[0] + 2., box_location[1] - text_size_lwir[1]]
          textbox_location_lwir = [box_location[0], box_location[1] - text_size_lwir[1], box_location[0] + text_size_lwir[0] + 4.,box_location[1]]

          draw.rectangle(xy=textbox_location_vis, fill=label_color_map[det_labels[i]])
          draw.text(xy=text_location_vis, text='{:.4f}'.format(det_scores[i].item()), fill='white', font=font)
          
          draw_lwir.rectangle(xy=textbox_location_lwir, fill=label_color_map[det_labels[i]])
          draw_lwir.text(xy=text_location_lwir, text='{:.4f}'.format(det_scores[i].item()), fill='white', font=font)
      
      new_image = Image.new('RGB',(original_image.size[0], original_image.size[1]))
      new_image.paste(original_image,(0,0))
      new_image_lwir = Image.new('RGB',(original_image.size[0], original_image.size[1]))
      new_image_lwir.paste(original_lwir,(0,0))

      del draw
      del draw_lwir
    
      return new_image, new_image_lwir

def visualize(result_filename, vis_dir, fdz_case , vis_vid , lwir_vid , min_score):

    data_list = list()
    for line in open(result_filename):
        data_list.append(line.strip().split(','))
    data_list = np.array(data_list)

    input_size = config.test.input_size
    vis_test_file = vis_vid
    lwir_test_file = lwir_vid
   
    vid = cv2.VideoCapture(vis_test_file)
    lwir_vid = cv2.VideoCapture(lwir_test_file)
   
    fps = lwir_vid.get(cv2.CAP_PROP_FPS)
    print('fps = ',fps)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    idx = 0

    if fdz_case == 'original':
      out = cv2.VideoWriter('{}/{}_detection_result.mp4'.format(vis_dir,fdz_case),cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280,512))
    else:
      out = cv2.VideoWriter('{}/{}_detection_result.mp4'.format(vis_dir,fdz_case),cv2.VideoWriter_fourcc(*'mp4v'), fps, (640,512))
     

    args = config.args
    cond = args.dataset.OBJ_LOAD_CONDITIONS['test'] 

    # Load dataloader for Fusion Dead Zone experiment
    FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]
    img_transform = Compose(FDZ)
    img_co_transform = Compose([Resize(input_size), 
                                ToTensor()
                                ])

    while True:
			
        ret ,vis_frame = vid.read()
        lwir_ret,lwir_frame1 = lwir_vid.read()
    
        if not ret :
          break
        if not lwir_ret:
          break
        if vis_frame is None :
          break
        if lwir_frame1 is None:
          break
        

        ##From Numpy to PIL Image
       
        vis = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        vis = Image.fromarray(vis.astype(np.uint8))
        lwir = Image.fromarray(lwir_frame1.astype(np.uint8))
        
        width, height = lwir.size
 
        ##convert(3,h,w)->(1,h,w)
        lwir= Grayscale()(lwir)

         #test mode ##
        boxes_vis = [[0, 0, 0, 0, -1]]
        boxes_lwir = [[0, 0, 0, 0, -1]]
        boxes_vis = np.array(boxes_vis, dtype=np.float)
        boxes_lwir = np.array(boxes_lwir, dtype=np.float)
              

          ## Apply transforms
        if img_transform is not None:
              
          vis, lwir, boxes_vis , boxes_lwir, _ = img_transform(vis, lwir, boxes_vis, boxes_lwir)
          
        if img_co_transform is not None:
              
              pair = 1
              
              vis, lwir, boxes_vis, boxes_lwir, pair = img_co_transform(vis, lwir, boxes_vis, boxes_lwir, pair)                      
              
              
              if boxes_vis is None:
                  boxes = boxes_lwir
              elif boxes_lwir is None:
                  boxes = boxes_vis
              else : 

                  ## Pair Condition
                  ## RGB / Thermal
                  ##  1  /  0  = 1
                  ##  0  /  1  = 2
                  ##  1  /  1  = 3

                  if pair == 1 :
                      
                      if len(boxes_vis.shape) != 1 :
                          boxes_vis[1:,4] = 3
                      if len(boxes_lwir.shape) != 1 :
                          boxes_lwir[1:,4] = 3
                  else : 
                      if len(boxes_vis.shape) != 1 :
                          boxes_vis[1:,4] = 1
                      if len(boxes_lwir.shape) != 1 :
                          boxes_lwir[1:,4] = 2
                  
                  boxes = torch.cat((boxes_vis,boxes_lwir), dim=0)
                  boxes = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes.numpy()]))))   

          ## Set ignore flags
        ignore = torch.zeros( boxes.size(0), dtype=torch.bool)
                
        for ii, box in enumerate(boxes):
                          
              x = box[0] * width
              y = box[1] * height
              w = ( box[2] - box[0] ) * width
              h = ( box[3] - box[1] ) * height

              if  x < cond['xRng'][0] or \
                  y < cond['xRng'][0] or \
                  x+w > cond['xRng'][1] or \
                  y+h > cond['xRng'][1] or \
                  w < cond['wRng'][0] or \
                  w > cond['wRng'][1] or \
                  h < cond['hRng'][0] or \
                  h > cond['hRng'][1]:

                  ignore[ii] = 1
          
        boxes[ignore, 4] = -1
          
        labels = boxes[:,4]
        boxes_t = boxes[:,0:4] 

        detection = data_list[data_list[:,0] == str(idx+1)].astype(float)
        idx += 4
     
        #detection result
        vis, lwir = detect(vis, lwir, detection, min_score)
        
        if fdz_case == 'original':
            img1 = vis.copy()
            img1 = np.array(img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = lwir.copy()
            img2 = np.array(img2)

            new_images = np.concatenate((img1, img2), axis=1)
        
        elif fdz_case == 'blackout_r':
            new_images = lwir.copy()
            new_images = np.array(new_images)
            
        elif fdz_case == 'blackout_t':
            new_images = vis.copy()
            new_images = np.array(new_images)
            new_images = cv2.cvtColor(new_images, cv2.COLOR_BGR2RGB)
            
        out.write(new_images)   
    out.release()
    lwir_vid.release()
    vid.release()
    cv2.destroyAllWindows()
    print('video saved!')



