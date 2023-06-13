---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
import shutil 
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchmetrics.classification import BinaryAccuracy

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import Trainer, seed_everything

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
```

```python
seed_everything(42, workers=True)
```

```python
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

class_names = ['chihuahua', 'muffin']
dataroot = '../data/muffin-vs-chihuahua-image-classification'
```

<!-- #region tags=[] -->
## Create validation set
<!-- #endregion -->

```python tags=[]
for class_name in class_names:
    source_dir = os.path.join(dataroot, test_dir, class_name)
    dest_dir = os.path.join(dataroot, val_dir, class_name)
    os.makedirs(dest_dir, exist_ok=True)  
    
    _, val_imgs = train_test_split(os.listdir(source_dir), test_size=0.5, random_state=42)
    for i, file_name in enumerate(tqdm(val_imgs)):
        shutil.move(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
```

# View dataset

```python
def print_len_in_folder(folder, title, classes=[]):
    print(f'{title} size:')
    for class_name in classes:
        print(f'{class_name} -', len(os.listdir(os.path.join(folder, class_name))))
```

```python
print_len_in_folder(os.path.join(dataroot, train_dir), "Train",class_names)
print_len_in_folder(os.path.join(dataroot, test_dir), "Test",class_names)
print_len_in_folder(os.path.join(dataroot, val_dir), "Val",class_names)
```

```python
view_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    
])
BATCH_SIZE = 9
view_dataset = ImageFolder(root = os.path.join(dataroot, 'train'), transform=view_transform)
view_loader = DataLoader(view_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
```

```python
def imshow(img,title):
    img = img.detach().cpu().numpy()
    plt.figure(figsize=(32,10))
    plt.imshow(img.transpose(1,2,0))
    plt.axis('off')
    plt.title(title)
    plt.show()
    
def imglabel(label):
    return 'Muffin' if label else 'Chihuahua'
    
def plot_batch(dataloader= None, image=None, label = None):
    if dataloader:
        image, label = next(iter(dataloader))
    assert len(image)>9
    plt.figure(figsize=(8,8))
    for i in range(1, 10):
        plt.subplot(3,3,i)
        plt.tight_layout()
        plt.imshow(image[i-1].permute(1, 2, 0))
        plt.xlabel(imglabel(label[i-1].item()), fontsize=10)
```

```python
plot_batch(dataloader = view_loader)
```

# Train model

```python
BATCH_SIZE = 64
```

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

```python
train_dataset = ImageFolder(root = os.path.join(dataroot, train_dir), transform=train_transform)
test_dataset = ImageFolder(root = os.path.join(dataroot, test_dir), transform=test_transform)
val_dataset = ImageFolder(root = os.path.join(dataroot, val_dir), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
```

```python
class TransferResNet(pl.LightningModule):
    def __init__(self, ):
        super().__init__()
        
        self.train_acc = BinaryAccuracy() 
        self.valid_acc = BinaryAccuracy() 
        self.test_acc = BinaryAccuracy()     
        
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential( nn.Linear(num_filters, num_filters),
                        nn.ReLU(),  
                        nn.Linear(num_filters, 1),
                        nn.Sigmoid())
        
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        
    def _predict(self, x):
        representations = self.feature_extractor(x).flatten(1)
        return self.classifier(representations)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self._predict(x))
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log("train_loss", loss)
        self.train_acc.update(y_hat, y) 
        return loss
    
    def on_train_epoch_end(self): 
        self.log("train_acc", self.train_acc.compute()) 
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self._predict(x))
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log("test_loss", loss)
        self.test_acc.update(y_hat, y) 
        return loss
    
    def on_test_epoch_end(self): 
        self.log("test_acc", self.test_acc.compute()) 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self._predict(x))
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log("valid_loss", loss)
        self.valid_acc.update(y_hat, y) 
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_acc", self.valid_acc.compute()) 
    
    def forward(self, x):
        return self._predict(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

```python
model = TransferResNet()
optimizer = model.configure_optimizers()

trainer = Trainer(max_epochs=10, check_val_every_n_epoch=1,
                  callbacks=[EarlyStopping(monitor="valid_loss", mode="min")])
trainer.fit(model, train_loader, val_loader)
```

```python
trainer.test(dataloaders = test_loader)
```

```python
t = Variable(torch.Tensor([0.5]))  # threshold
```

```python
image, label = next(iter(test_loader))
model.freeze()
predictions = model(image)
out = (predictions > t)
plot_batch(image = image, label = out)
```

```python
# View metrics
#tensorboard --logdir lightning_logs/ 
```
