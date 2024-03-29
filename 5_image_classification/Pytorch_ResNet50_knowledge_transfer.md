---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_accuracy

from torch.utils.data import DataLoader
from torch.autograd import Variable

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
dataroot = '../data/muffin-vs-chihuahua-image-classification'
train_dir = 'train'
test_dir = 'test'

subsets = [train_dir,test_dir ]
class_names = ['chihuahua', 'muffin']
```

# View dataset

```python
def print_dataset_len(dataroot, class_name, subsets=None):
    existing_sets = subsets or []
    print(f'{class_name}:')
    lengths = {s:len(os.listdir(os.path.join(dataroot, s, class_name))) for s in existing_sets}
    total_sum = sum(lengths.values())
    if total_sum == 0:
        print("Not found")
    else:
        for name, l in lengths.items():
            print(f'{name} - {l} : {l/total_sum*100:.2f}%')
```

```python
print("Dataset lenght by class and split:")
for class_name in class_names:
    print_dataset_len(dataroot, class_name, subsets)
```

```python
VIEW_BATCH_SIZE = 9

view_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    
])
view_dataset = ImageFolder(root = os.path.join(dataroot, train_dir), transform=view_transform)
view_loader = DataLoader(view_dataset, batch_size=VIEW_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
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
class DogVsMuffinDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = ".", batch_size = 32, num_workers = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage = None):
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
        train_folder = ImageFolder(root = os.path.join(self.data_dir, train_dir), transform=train_transform)
        self.train, self.validate = torch.utils.data.random_split(train_folder, [0.8, 0.2])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        self.test =  ImageFolder(root = os.path.join(self.data_dir, test_dir), transform=test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,
                           shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
```

```python
class TransferResNet(pl.LightningModule):
    def __init__(self, ):
        super().__init__()
        
        backbone = models.resnet50(weights='DEFAULT')
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(num_filters, num_filters),
                        nn.ReLU(),  
                        nn.Linear(num_filters, 1))
        
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    
    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        classifier_scores = self.classifier(representations)
        return classifier_scores

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self.forward(x))
        loss = self.loss(y_hat, y)
        acc = binary_accuracy(y_hat, y)
        
        metrics = {
            'train_loss': loss,
            'train_acc': acc
            } 
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self.forward(x))
        loss = self.loss(y_hat, y)
        acc = binary_accuracy(y_hat, y)
        metrics = {
            'test_loss': loss,
            'test_acc': acc
            } 
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self.forward(x))
        loss = self.loss(y_hat, y)
        acc = binary_accuracy(y_hat, y)
        
        metrics = {
            'val_loss': loss,
            'val_accuracy': acc
            } 
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

```python
model = TransferResNet()
data_module = DogVsMuffinDataModule(dataroot, BATCH_SIZE, num_workers = 4)
optimizer = model.configure_optimizers()

trainer = Trainer(max_epochs=20, check_val_every_n_epoch=1,
                  default_root_dir="./muffin-or-dog_lightning",
                  callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode='min')])
trainer.fit(model, datamodule = data_module)
```

```python
trainer.test(datamodule = data_module)
```

# Visualize results

```python
t = Variable(torch.Tensor([0.5]))  # threshold
sig = nn.Sigmoid()
```

```python
image, label = next(iter(data_module.test_dataloader()))
model.eval()
predictions =  sig(model(image))
out = (predictions > t)
plot_batch(image = image, label = out)
```

## Show wrongly classified

```python
loader = data_module.test_dataloader()
```

```python
w_image =[]
w_labels =[]
model.eval()

test_loader_iter = iter(loader)
try:
    while len(w_image) <= VIEW_BATCH_SIZE:
        image, label = next(test_loader_iter)
        predictions = sig(model(image))
        out = (predictions > t).type(torch.int64).flatten()
        wrong_ind = (out != label)
        w_image.extend(image[wrong_ind,:,:,:])
        w_labels.extend(out[wrong_ind])
except StopIteration:
    pass
print(f'Found {len(w_image)} wrongly classified examples')
```

```python
plot_batch(image = w_image, label = w_labels)
```
