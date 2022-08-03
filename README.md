# Fusion-Model-Architecture
Multispectral pedestrian detection

## Prerequisites

- Ubuntu 18.04
- Python 3.7
- Pytorch 1.6.0
- Torchvision 0.7.0
- CUDA 10.1
- requirements.txt

## Getting Started

### Git Clone 
```
git clone hhttps://github.com/nourhenesarraj/Fusion-Model-Architecture.git
```
```
cd Fusion-Model-Architecture
```

## Dataset

For multispectral pedestrian detection, we train and test the  model on our [Robot dataset](https://drive.google.com/drive/folders/1hfP4pmSKW5jtWkxXdeIOG4RVswVKG-BU?usp=sharing), you should first download the dataset. By default, we assume the dataset is stored in `data/ARCNNDataset`. Please see more details below.

It is recommended to symlink the dataset root to data/ARCNNDataset.

```
+-- src
|   +-- imageSets
|   |   +-- train-all-02.txt
|   |   +-- test-all-20.txt
+-- data
|   +-- data/ARCNNDataset
|   |   +-- annotations
|   |   |   +-- set00
|   |   |   |   +-- V000
|   |   |   |   |   +-- lwir
|   |   |   |   |   |   +-- frame0.txt
|   |   |   |   |   +-- visible
|   |   |   |   |   |   +-- frame0.txt
|   |   +-- images
|   |   |   +-- set00
|   |   |   |   +-- V000
|   |   |   |   |   +-- lwir
|   |   |   |   |   |   +-- frame0.jpg
|   |   |   |   |   +-- visible
|   |   |   |   |   |   +-- frame0.jpg
|   |   |   +-- set01
|   |   |   |   +-- V000
|   |   |   |   |   +-- lwir
|   |   |   |   |   |   +-- frame0.jpg
|   |   |   |   |   +-- visible
|   |   |   |   |   |   +-- frame0.jpg
|   |   +-- splits
|   |   |   +-- trainval.txt
|   |   |   +-- test.txt

```

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

## Training and Evaluation

If you want to change default parameters, you can modify them in the module `src/config.py`.

### Train
Please, refer to the following code to train and evaluate the model.
```
cd src
python train_eval.py
```

### Pretrained Model
If you want to skip the training process, download the pre-trained model and place it in the directory `pretrained/`.

- [Pretrained Model](https://drive.google.com/file/d/10aL_I_8QR9UCq8tXLge7pgp2GWst5neI/view?usp=sharing)

### Inference

Try below command to get inference from pretrained model

```bash
$ cd src
$ python inference.py --FDZ original --model-path ../pretrained/best_checkpoint.pth.tar
```
++ If you want to visualize the results, try addding the `--vis` argument. Like below

```bash
$ python inference.py --FDZ original --model-path ../pretrained/best_checkpoint.pth.tar --vis
```
Visualization results are stored in 'result/vis'.

### Convert model to onnx

```bash
$ python onnx_model.py --model-path ../pretrained/best_checkpoint.pth.tar
```
### Inference with onnx model
Install onnx-runtime 

```bash
pip install onnxruntime-gpu
```

```bash
$ python onnx_inference.py --FDZ original --model-path ../pretrained/best_checkpoint.pth.tar --vis
```

### Demo video

```bash
!python demo_video.py --FDZ blackout_r --model_path ../pretrained/best_checkpoint.pth.tar --vis
```

Note: 
FDZ arg could be : - original (for fusion) or blackout_r (blackout for Visible image) or blackout_t (Blackout for Thermal image) 

