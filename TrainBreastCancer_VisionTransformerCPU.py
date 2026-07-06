import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision import datasets 

import time

DirnameTrain="Dir_BreastCancer_VisionTransformerCPU/train"
DirnameVal="Dir_BreastCancer_VisionTransformerCPU/val"
DirnameTest="Dir_BreastCancer_VisionTransformerCPU/test"

# 1. Configuración de hiperparámetros para CPU estricta
device = torch.device("cpu")
BATCH_SIZE = 64          
            
EPOCHS = 2500 # no multiprocess

LEARNING_RATE = 0.0001 #MOD

# Hiperparámetros de la arquitectura ViT Mini

PATCH_SIZE = 16   # Parches de 16x16 for images  128x128 (64 parches in total)

IMAGE_SIZE = 128 #MOD

NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
INPUT_CHANNELS = 3
D_MODEL = 64             # Dimensión oculta compacta
NUM_HEADS = 4            
NUM_LAYERS = 2           
MLP_DIM = 128            

NUM_CLASSES = 2 # no cancer, cancer

print(f"Device: {device} (train sequential,  without threads")


# 2. Preparación de los datos (BreastCancer)
transform_train = transforms.Compose([
    
    transforms.RandomCrop(128, padding=4), # MOD
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_data = datasets.ImageFolder( DirnameTrain, transform=transform_train)
valid_data = datasets.ImageFolder( DirnameVal, transform=transform_train)
test_data = datasets.ImageFolder( DirnameTest, transform=transform_test)

#train_data, valid_data = torch.utils.data.random_split(train_data, [0.7, 0.3]) # MOD

# IMPORTANT num_workers=0 means there is no multiprocess, that gives problems 
trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
validloader = torch.utils.data.DataLoader(valid_data,batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
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

# 4. Inicialización
model = VisionTransformerCPU().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 5. Bucle de entrenamiento monohilo
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(trainloader)}] | Pérdida: {running_loss/(batch_idx+1):.4f}")

    epoch_time = time.time() - start_time
    train_acc = 100. * correct / total
    print(f"--- End Époch {epoch+1} | Accuracy: {train_acc:.2f}% | Time epoch : {epoch_time:.2f}s ---")
#model.save_pretrained("data")
torch.save(model.state_dict(), f'mi_modelo_vit3.pth', _use_new_zipfile_serialization=False)
# 6. Evaluación final en Test
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        print(outputs)
        print("*****")
        print(targets)
        print("+++++")
        _, predicted = outputs.max(1)
        print(predicted)
        print("-------")
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

print(f"\n✅ Precisión final en TEST (sin multiproceso): {100. * test_correct / test_total:.2f}%")

