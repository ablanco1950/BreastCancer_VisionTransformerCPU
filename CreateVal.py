import os
import shutil
output_dir="Dir_BreastCancer_VisionTransformerCPU\\val"

if  os.path.exists(output_dir):shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + "\\0")
os.mkdir(output_dir + "\\1")

import cv2

import numpy as np

# Abrir el archivo .npz
with np.load('breastmnist_128.npz') as data:
    # 1. Ver los nombres de las variables / arreglos guardados
    #print(data.files)
    
    # 2. Acceder a un arreglo específico usando su nombre
    val_images = data['val_images']
    #print(mi_matriz)
    print(val_images.shape)
    
with np.load('breastmnist_128.npz') as data:
    # 1. Ver los nombres de las variables / arreglos guardados
   # print(data.files)
    
    # 2. Acceder a un arreglo específico usando su nombre
    val_labels = data['val_labels']
    #print(mi_matriz)
    print(val_labels.shape)
    #print(val_labels)
    
NameImage=0
for i in range(len(val_images)):
    NameImage=NameImage + 1
    if val_labels[i] == [1]:
        cv2.imwrite(output_dir + "\\1\\" + str(NameImage)+ ".jpg" , val_images[i])
    else:
        cv2.imwrite(output_dir + "\\0\\" + str(NameImage)+ ".jpg" , val_images[i])
