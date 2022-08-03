import os
from os.path import join as opj
from pathlib import Path
from pickle import STRING
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
import time
import config

import numpy as np
import glob
from google.colab.patches import cv2_imshow

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.transforms import FusionDeadZone, Compose, Resize, ToTensor , Normalize
from torchvision.transforms import Grayscale

from PIL import Image, ImageDraw, ImageFont , ImageOps
import cv2
import imageio

from datasets import KAISTPed
from utils.transforms import FusionDeadZone
from utils.evaluation_script import evaluate
from vis_video import detect , visualize

from model import SSD300
import onnxruntime as ort


def val_epoch(model: SSD300, input_size: Tuple,fdz_case: str, onnx_path : str , vis_vid : str , lwir_vid : str,  min_score: float = 0.1) -> Dict:

    model.eval()

    #onnx session
    provider =['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=provider)
    
    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)

    device = next(model.parameters()).device
    results = dict()
    ###transformations 
    FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]

    img_transform = Compose(FDZ)
    img_co_transform = Compose([Resize(input_size), 
                                ToTensor(),
                                Normalize([0.3465,  0.3219,  0.2842],[0.2358, 0.2265, 0.2274],'R'),
                                Normalize([0.1598],[0.0813],'T')
                                ])
    args = config.args
    cond = args.dataset.OBJ_LOAD_CONDITIONS['test']                           
    ##videos ####
    vis_test_file = vis_vid
    lwir_test_file = lwir_vid
  
    vid = cv2.VideoCapture(vis_test_file)
    lwir_vid = cv2.VideoCapture(lwir_test_file)

    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    i = 0
    idx = 0

    while True:
			
      ret , vis_frame = vid.read()
      lwir_ret, lwir_frame1 = lwir_vid.read()
      
      if not ret :
        break
      if not lwir_ret:
        break
        
      vis = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
      vis = Image.fromarray(vis_frame.astype(np.uint8))
      lwir = Image.fromarray(lwir_frame1.astype(np.uint8))
      lwir= Grayscale()(lwir)
      width, height = lwir.size

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
     
      ##new shape from (C,H,W) -> (B,C,H,W)
      vis = torch.unsqueeze(vis, 0) 
      lwir = torch.unsqueeze(lwir, 0) 
      img_vis = vis.numpy()
      img_lwir = lwir.numpy()
 
      ort_inputs = {
              "image_vis": img_vis ,
              "image_lwir":img_lwir
              }

      indices = [torch.tensor([i], dtype=torch.int32),torch.tensor([i+1], dtype=torch.int32),torch.tensor([i+2], dtype=torch.int32),torch.tensor([i+3], dtype=torch.int32)]
      i += 4
      
      # Forward prop.
      ort_outs = ort_session.run(None,ort_inputs)

      predicted_locs = torch.from_numpy(ort_outs[0]).to(device)
      predicted_scores = torch.from_numpy(ort_outs[1]).to(device)

      #  Detect objects in SSD output
      detections = model.module.detect_objects(predicted_locs, predicted_scores,
                                                     min_score=min_score, max_overlap=0.425, top_k=200)
            
      det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]
    
      for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, indices):
                boxes_np = boxes_t.detach().cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.detach().cpu().numpy().mean(axis=1).reshape(-1, 1)
                labels_np = labels_t.detach().cpu().numpy().reshape(-1, 1)

                xyxy_np = boxes_np * xyxy_scaler_np
                xywh_np = xyxy_np
                xywh_np[:, 2] -= xywh_np[:, 0]
                xywh_np[:, 3] -= xywh_np[:, 1]

                results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])
      
    return results
def save_results(results: Dict, result_filename: str):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    Parameters
    ----------
    results: Dict
        Detection results for each image_id: {image_id: box_xywh + score}
    result_filename: str
        Full path of result file name

    """
    if not result_filename.endswith('.txt'):
        result_filename += '.txt'
   
    with open(result_filename, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x, y, w, h, score in detections:
                
                f.write(f'{image_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score:.8f}\n')
   
def run_inference(onnx_path: str, fdz_case: str , vis_vid : str, lwir_vid : str , model_path : str) -> Dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model']
    model = model.to(device)

    model = nn.DataParallel(model)

    input_size = config.test.input_size

    results = val_epoch(model,input_size,fdz_case, onnx_path , vis_vid , lwir_vid)

    return results


if __name__ == '__main__':

    FDZ_list = FusionDeadZone._FDZ_list

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--FDZ', default='original', type=str, choices=FDZ_list,
                        help='Setting for the "Fusion Dead Zone" experiment. e.g. {}'.format(', '.join(FDZ_list)))
    parser.add_argument('--onnx', required=True, type=str,
                        help='Pretrained model.')
    parser.add_argument('--result-dir', type=str, default='../results_videos_detection',
                        help='Save result directory')
    parser.add_argument('--videolwir', type=str, default='/content/drive/MyDrive/robot-videos*/lwir_vid_1.mp4',
                        help='lwir video')
    parser.add_argument('--videovis', type=str, default='/content/drive/MyDrive/robot-videos*/vis_vid_1.mp4',
                        help='visible video')                   
    parser.add_argument('--vis', action='store_true', 
                        help='Visualizing the results')

    arguments = parser.parse_args()

    print(arguments)
    start = time.time()
    
    # Arguments
    fdz_case = arguments.FDZ.lower()
    model_path_name = 'checkpoint_ssd300_onnx'
    model_path = './jobs/2022-05-23_18h32m_/checkpoint_ssd300.pth.tar040'
    vis_vid = arguments.videovis
    lwir_vid = arguments.videolwir 

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True)
    result_filename = opj(arguments.result_dir,  f'{fdz_case}_{model_path_name}_TEST_det')

    # Run inference
    results = run_inference(arguments.onnx, fdz_case, vis_vid, lwir_vid, model_path)

    save_results(results, result_filename)

    if arguments.vis:
        vis_dir = opj(arguments.result_dir, 'vis', model_path_name, fdz_case)
        os.makedirs(vis_dir, exist_ok=True)
        if fdz_case in ['blackout_r' , 'original'] :
          visualize(result_filename + '.txt', vis_dir, fdz_case , vis_vid , lwir_vid, 0.3)
        else:
          print('score 0.2')
          visualize(result_filename + '.txt', vis_dir, fdz_case , vis_vid , lwir_vid, 0.2)
    end = time.time()
    print(" execution took: %2.5f sec" % (end - start))
    
    