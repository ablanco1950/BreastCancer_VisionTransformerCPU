import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision import datasets # MOD

dirname = 'Dir_BreastCancer_VisionTransformerCPU/test'
TrainedModel='mi_modelo_vit.pth'
classes=["0 No Cancer", "1 Cancer"]


import time

# 1. Configuración de hiperparámetros para CPU estricta
device = torch.device("cpu")
BATCH_SIZE = 64          

# Hiperparámetros de la arquitectura ViT Mini

PATCH_SIZE = 16            # Parches de 16x16 para imágenes de 128x128 (64 parches en total)

IMAGE_SIZE = 128 

NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
INPUT_CHANNELS = 3
D_MODEL = 64             # Dimensión oculta compacta
NUM_HEADS = 4            
NUM_LAYERS = 2           
MLP_DIM = 128            

NUM_CLASSES = 2 # 0 no cancer, 1 cancer


import os
import re
def loadimagesTest(dirname):
    imgpath = dirname

    print("Reading imagenes from ",imgpath)

    TabDirName=[]
    for root, dirnames, filenames in os.walk(imgpath):
         for dirname in dirnames:  
             #print(dirname)
             
             TabDirName.append(dirname)

    TotImages=0
   
    TotImagesValid=0
    TabImagePath = []
    NameImages=[]
    images=[]
    Y=[]
    Y_numerical=[]
    
    for i in range(len(TabDirName)):
    
        imgpath1=imgpath + "\\" + str(TabDirName[i])

        #print(imgpath1)
        target = TabDirName[i]

        
        # https://stackoverflow.com/questions/62137343/how-to-get-full-path-with-os-walk-function-in-python
        for root, dirnames, filenames in os.walk(imgpath1):
            for filename in filenames:  
                
                filepath = os.path.join(root, filename)
                
                TabImagePath.append(filepath)
                NameImages.append(filename)
                #Y.append(TabDirName[i])
                Y.append(TabDirName[i])
                #Y_numerical.append(index)
                
                TotImages+=1
                
                                
                
    print( " Total images to test "  + str(TotImages))
    
    #for i in range(len(Y)):
    #    print(NameImages[i] + " class " + Y[i])
    return TabImagePath, Y,  NameImages, images


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


test_data = datasets.ImageFolder( dirname, transform=transform_test)

# crucial num_workers=0 to avoid multiprocessing with is problematic working with cpu
testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. Arquitectura Vision Transformer (ViT) Modificada
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, d_model):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # [B, d_model, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [B, Num_Patches, d_model]
        return x

class VisionTransformerCPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(INPUT_CHANNELS, PATCH_SIZE, D_MODEL)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embedding = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, D_MODEL))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NUM_HEADS, 
            dim_feedforward=MLP_DIM, 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, NUM_CLASSES)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

TabImagePath, Y,  NameImages, images=  loadimagesTest(dirname)  

# 4. Inicialización
model = VisionTransformerCPU().to(device)



# 2. Carga el diccionario de pesos (state_dict) desde el archivo .pth
state_dict = torch.load(TrainedModel, weights_only=True)

# 3. Asigna los pesos cargados a la estructura de tu modelo
model.load_state_dict(state_dict)
model.eval()

test_correct = 0
test_total = 0

Cont=0
ContHits=0
ContFailures= 0

TabPredicted=[]

import numpy 
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        #print(outputs)
        outputs_arr=outputs.numpy()
        #outputs_arr=outputs_arr[0]
        #print(outputs_arr)
        for i in range (len(outputs_arr)):
            if outputs_arr[i][0] > outputs_arr[i][1]:
                target_predicted = "0"
                
            else:
                target_predicted = "1"
            TabPredicted.append(target_predicted)    
            if target_predicted== Y[Cont]:
                print( "HIT " + NameImages[Cont] + " class predicted = " + target_predicted)
                ContHits=ContHits+1
            else:    
                print( "FAILURE " + NameImages[Cont] + " class predicted = " + target_predicted + " true class = " + Y[Cont])
                ContFailures=ContFailures+1

            Cont= Cont + 1

print("")
print("")
print("Total Hits= " + str(ContHits))
print("Total Failures= " + str(ContFailures))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import  ConfusionMatrixDisplay
# Compute confusion matrix
cm = confusion_matrix(Y, TabPredicted)
print("")
# Optional: print raw matrix
print("Confusion Matrix (raw values):\n", cm)

#def find_classes(dir):
#    classes = os.listdir(dir)
#    classes.sort()
#    class_to_idx = {classes[i]: i for i in range(len(classes))}
#    return classes, class_to_idx

#classes, c_to_idx = find_classes(dirname)

#print(classes, c_to_idx)

import matplotlib.pyplot as plt
# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Breast Cancer Classification")
plt.show()
print("")


