import os
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
import config
import numpy as np

import onnxruntime as ort

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import KAISTPed
from utils.transforms import FusionDeadZone
from utils.evaluation_script import evaluate
from vis import visualize

from model import SSD300




def val_epoch(model: SSD300,dataloader: DataLoader, input_size: Tuple, onnx_path : str , min_score: float = 0.1) -> Dict:
    """Validate the model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    input_size: Tuple
        A tuple of (height, width) for input image to restore bounding box from the raw prediction
    min_score: float
        Detection score threshold, i.e. low-confidence detections(< min_score) will be discarded

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

  
    model.eval()
    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = dict()
    provider =[ 'CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path,providers=provider)
    with torch.no_grad():
        for i, blob in enumerate(tqdm(dataloader, desc='Evaluating')):
            image_vis, image_lwir, boxes, labels, indices = blob
            image_lwir = image_lwir.resize_(1, 1, 512, 640)
            image_vis = image_vis.resize_(1, 3, 512, 640)
          
            image_vis = image_vis.numpy()
            image_lwir = image_lwir.numpy()
            
            ort_inputs = {
              "image_vis": image_vis ,
              "image_lwir":image_lwir
              }
            # Forward prop.
            ort_outs = ort_session.run(None,ort_inputs)

            predicted_locs = torch.from_numpy(ort_outs[0]).to(device)
            predicted_scores = torch.from_numpy(ort_outs[1]).to(device)

            # Detect objects in SSD output
            detections = model.module.detect_objects(predicted_locs, predicted_scores,
                                                     min_score=min_score, max_overlap=0.425, top_k=200)
            
            det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]

            for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, indices):
                boxes_np = boxes_t.cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)

                # TODO(sohwang): check if labels are required
                # labels_np = labels_t.cpu().numpy().reshape(-1, 1)

                xyxy_np = boxes_np * xyxy_scaler_np
                xywh_np = xyxy_np
                xywh_np[:, 2] -= xywh_np[:, 0]
                xywh_np[:, 3] -= xywh_np[:, 1]

                results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])
    return results


def run_inference(onnx_path : str, fdz_case: str, model_path: str) -> Dict:
    """Load model and run inference

    Load pretrained model and run inference on KAIST dataset with FDZ setting.

    Parameters
    ----------
    model_path: str
        Full path of pytorch model
    fdz_case: str
        Fusion dead zone case defined in utils/transforms.py:FusionDeadZone

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model']
    model = model.to(device)

    model = nn.DataParallel(model)

    input_size = config.test.input_size
    

    # Load dataloader for Fusion Dead Zone experiment
    FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]
    config.test.img_transform.add(FDZ)

    args = config.args
 
    batch_size = config.test.batch_size * torch.cuda.device_count()
    
    test_dataset = KAISTPed(args, condition="test")

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.dataset.workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)

    results = val_epoch(model, test_loader, input_size, onnx_path)
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


if __name__ == '__main__':

    FDZ_list = FusionDeadZone._FDZ_list

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--FDZ', default='original', type=str, choices=FDZ_list,
                        help='Setting for the "Fusion Dead Zone" experiment. e.g. {}'.format(', '.join(FDZ_list)))
    parser.add_argument('--onnx', required=True, type=str,
                        help='Pretrained model for evaluation.')
    parser.add_argument('--result-dir', type=str, default='./onnx-results',
                        help='Save result directory')
    parser.add_argument('--vis', action='store_true', 
                        help='Visualizing the results')
    arguments = parser.parse_args()

    print(arguments)

    fdz_case = arguments.FDZ.lower()
    model_path_name = 'checkpoint_ssd300_onnx'
    model_path = './jobs/2022-05-23_18h32m_/checkpoint_ssd300.pth.tar040'

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True)
    result_filename = opj(arguments.result_dir,  f'{fdz_case}_{model_path_name}_TEST_det')

    # Run inference
    results = run_inference(arguments.onnx, fdz_case, model_path)

    # #Save results
    save_results(results, result_filename)

    # # Eval results
    # # phase = "Multispectral"
    # # evaluate(config.PATH.JSON_GT_FILE, result_filename + '.txt', phase) 
    
    # Visualizing
    if arguments.vis:
        vis_dir = opj(arguments.result_dir, 'vis', model_path_name, fdz_case)
        os.makedirs(vis_dir, exist_ok=True)
        visualize(result_filename + '.txt', vis_dir, fdz_case)
