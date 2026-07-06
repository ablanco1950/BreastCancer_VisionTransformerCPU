# BreastCancer_VisionTransformerCPU
Classification of breast cancer (among 2 classes: 0 No cancer, 1 Cancer ) using the file https://zenodo.org/records/10519652 dataset breastmnist_128.npz and the vision transformer model with cpu and no multprocess 

Requirements:
The model has been trained, with 2500 epoch, in  a laptot wit 16Gb in several hours
INSTALLATION:
A requirements.txt file is attached.
Or you can install the following modules one by one if they are not already installed
>python pip-script.py install torch
python pip-script.py install torchvision
>python pip-script.py install scikit-learn
python pip-script.py install packaging
python pip-script.py install pyparsing
python pip-script.py install cycler
python pip-script.py install python-dateutil
python pip-script.py install kiwisolver
python pip-script.py install importlib_resources
python pip-script.py install opencv-python opencv-contrib-python

Download all the files that accompany this project in a single folder.
Download from https://zenodo.org/records/10519652 the dataset breastmnist_128.npz  and put in the project folder.
Create and fill the structure that needs the project, executing:
python CreateTrain.py
python CreateVal.py
python CreateTest.py

EVALUATION
python EvaluateBreastCancer_VisionTransformerCPU.py
The model displays a list of images, by name,  whose classes were correctly predicted and those that were not, resulting in an accuracy of = 85% (hit predictions / (hit predictions + error predictions)).
TRAINING
The mi_modelo_vit model was obtained through training
python TrainBreastCancer_VisionTransformerCPU
With 2500 epoch,  runned in a laptot wit 16Gb in several hours
REFERENCES:

https://zenodo.org/records/10519652

https://github.com/ablanco1950/BreastCancerClassification_Yolo26

https://github.com/ablanco1950/SkinLessionClassification_Yolo26

https://github.com/ablanco1950/SkinLesionDetection_Resnet_Pytorch

https://umeshk1255.medium.com/breast-cancer-detection-using-ai-understanding-convolutional-neural-networks-168d91f5cafc

Citation

Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, & Bingbing Ni. (2024). [MedMNIST+] 18x Standardized Datasets for 2D and 3D Biomedical Image Classification with Multiple Size Options: 28 (MNIST-Like), 64, 128, and 224 (3.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10519652 Style APA
