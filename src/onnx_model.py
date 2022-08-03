import torch
from torch.utils.data import DataLoader
import config
from datasets import KAISTPed
from tqdm import tqdm
import argparse
from os.path import join as opj


parser = argparse.ArgumentParser(description='Process some values.')
  
parser.add_argument('--model-path', required=True, type=str,
                        help='Pretrained model for conversion to onnx.')
parser.add_argument('--result-dir', type=str, default='./jobs',
                        help='Save result directory')

arguments = parser.parse_args()

print(arguments)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = arguments.model_path

model = torch.load(file)['model']
model = model.to(device)

args = config.args
train_conf = config.train
phase = "Multispectral"

train_dataset = KAISTPed(args, condition="train")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=1,
                                              num_workers=args.dataset.workers,
                                              collate_fn=train_dataset.collate_fn,
                                              pin_memory=True)
with torch.no_grad():
        for batch_idx, (image_vis, image_lwir, boxes, labels, _) in enumerate(train_loader):
            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

save_name = opj(arguments.result_dir,  f'onnx_SSD300_model.onnx')

torch.onnx.export(
        model, #model
        (
          image_vis,
	        image_lwir
        ),  # model input (or a tuple for multiple inputs)
        save_name, 
        export_params=True,
        input_names=["image_vis", "image_lwir"],  # the model's input names
        output_names=["predicted_locs","predicted_scores"],
        verbose=True
      
    )

print("Model converted successfully. ONNX format model is at %s: ", save_name)
