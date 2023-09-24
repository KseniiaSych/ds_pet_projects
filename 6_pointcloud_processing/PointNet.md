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

Code is borrowed from from [Nikita Karaev's](https://github.com/nikitakaraevv) [PointNet implementation](https://github.com/nikitakaraevv/pointnet) with eventual changes in order to play around with the model. 

```python
import os
from pathlib  import Path
import itertools
import time
import csv

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import lightning.pytorch as pl
from lightning.pytorch import Trainer

import torch.nn as nn
import torch.nn.functional as F

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

import math, random
```

# Visualize

```python
path = Path("../data/ModelNet40")

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)}
print("Classess: ")
classes
```

```python
cls_size = pd.DataFrame.from_dict(
    {folder: sum([len(os.listdir(path/folder/subfold)) for subfold in os.listdir(path/folder)]) for folder in folders},
     orient='index',
    columns = ["cls_len"])
```

```python
cls_size
```

```python
len_proportion = cls_size.cls_len/sum(cls_size.cls_len)*100
len_proportion
```

```python
plot = len_proportion.plot.pie(figsize=(8, 8),rotatelabels=True, autopct='%1.1f%%', pctdistance=0.85)
plot.axes.get_yaxis().set_visible(False)
```

```python
def read_off(file):
    header = file.readline().strip()
    if not header.startswith('OFF'):
        raise ValueError('Not a valid OFF header')
    if len(header)>3:
        n_verts, n_faces, __ = tuple([int(s) for s in header[3:].split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces
```

```python
with open(path/"flower_pot/train/flower_pot_0001.off", 'r') as f:
    verts, faces = read_off(f)
```

```python
i,j,k = np.array(faces).T
x,y,z = np.array(verts).T
```

```python
len(x)
```

```python
def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig
```

```python
visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50, i=i,j=j,k=k)]).show()
```

```python
visualize_rotate([go.Scatter3d(x=x, y=y, z=z, mode='markers')]).show()
```

```python
def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
```

```python
pcshow(x,y,z)
```

# Transform

```python
class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points
```

```python
pointcloud = PointSampler(3000)((verts, faces))
```

```python
pcshow(*pointcloud.T)
```

# Normalize

```python
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
```

```python
norm_pointcloud = Normalize()(pointcloud)
```

```python
pcshow(*norm_pointcloud.T)
```

# Augmentations

```python
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandRotation_x(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[1,               0,                0],
                               [0, math.cos(theta), -math.sin(theta)],
                               [0, math.sin(theta),  math.cos(theta)]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandRotation_y(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), 0, math.sin(theta)],
                               [0,               1,               0],
                               [-math.sin(theta),0, math.cos(theta)]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud
```

```python
rot_pointcloud = RandRotation_z()(norm_pointcloud)
noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
```

```python
pcshow(*noisy_rot_pointcloud.T)
```

```python
rotx_pointcloud = RandRotation_x()(norm_pointcloud)
noisy_rotx_pointcloud = RandomNoise()(rotx_pointcloud)
pcshow(*noisy_rotx_pointcloud.T)
```

```python
roty_pointcloud = RandRotation_y()(norm_pointcloud)
noisy_roty_pointcloud = RandomNoise()(roty_pointcloud)
pcshow(*noisy_roty_pointcloud.T)
```

# ToTensor

```python
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)    
```

```python
ToTensor()(noisy_rot_pointcloud)
```

```python
def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])
```

# Dataset

```python
class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 
                'category': self.classes[category]}
```

```python
train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_x(),
                    RandRotation_y(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])
```

```python
train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)
```

```python
inv_classes = {i: cat for cat, i in train_ds.classes.items()};
inv_classes
```

```python
train_len = len(train_ds)
valid_len = len(valid_ds)
total = train_len + valid_len
n_classes =  len(train_ds.classes)
print('Train dataset size: ', train_len, ' - ', f'{train_len/total:.2f}')
print('Valid dataset size: ', valid_len, ' - ', f'{valid_len/total:.2f}')
print('Number of classes: ', n_classes)
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Example class: ', inv_classes[train_ds[0]['category']])
```

```python
batch_size = 128
```

```python
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size)
```

# Model

```python
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
       

    def forward(self, input):
      # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(pl.LightningModule):
    def __init__(self, classes = 40):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64
    
    def loss(self, outputs, labels, m3x3, m64x64, alpha = 0.0001):
        criterion = torch.nn.NLLLoss()
        bs=outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
        if outputs.is_cuda:
            id3x3=id3x3.cuda()
            id64x64=id64x64.cuda()
        diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
        diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
        return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
    
    def accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        return accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy(), normalize=True)
    
    def step(self, batch, batch_idx, metrics_names):
        inputs, labels = batch['pointcloud'].float(), batch['category']
        outputs, m3x3, m64x64 = self.forward(inputs.transpose(1,2))
        loss = self.loss(outputs, labels, m3x3, m64x64)
        acc = self.accuracy(outputs, labels)
        metrics = dict(zip(metrics_names, [loss, acc]))
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step( batch, batch_idx, ['train_loss', 'train_acc'])
    
    def test_step(self, batch, batch_idx):
        return self.step( batch, batch_idx, ['test_loss', 'test_acc'])
    
    def validation_step(self, batch, batch_idx):
        return self.step( batch, batch_idx, ['val_loss', 'val_acc'])
```

```python
model = PointNet(classes = n_classes)
optimizer = model.configure_optimizers()
trainer = Trainer(max_epochs=15, check_val_every_n_epoch=1, log_every_n_steps = 10)
trainer.fit(model,  train_loader, valid_loader)
```

```python
train(pointnet, train_loader, valid_loader,  save=False)
```

# Test

```python
pointnet =  PointNet.load_from_checkpoint("./pointnet_from_checkpoint/lightning_logs/version_1/checkpoints/epoch=5-step=462.ckpt")
pointnet.eval();
```

```python
pointnet
```

```python
all_preds = []
all_labels = []

all_preds_np = np.empty((0,), int)
all_labels_np = np.empty((0,), int)
with torch.no_grad():
    for i, data in enumerate(valid_loader):
        print('Batch [%4d / %4d]' % (i+1, len(valid_loader)))
                   
        inputs, labels = data['pointcloud'].float(), data['category']
        outputs, __, __ = pointnet(inputs.transpose(1,2).cuda())
        _, preds = torch.max(outputs.data, 1)
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        all_preds += list(preds_np)
        all_labels += list(labels_np)
        all_preds_np = np.append(all_preds_np, preds_np, axis=0)
        all_labels_np = np.append(all_labels_np, labels_np, axis=0)
```

```python
labels = list(inv_classes.keys())
accuracy = accuracy_score(all_labels_np, all_preds_np, normalize=True)
avg_precision = precision_score(all_labels_np, all_preds_np, average='macro', labels = labels )
avg_recall = recall_score(all_labels_np, all_preds_np, average='macro', labels = labels)
avg_f1 = f1_score(all_labels_np, all_preds_np, average='macro')

print(f'Overall accuracy - {accuracy:.4f}')
print(f'Mean average precision - {avg_precision:.4f}')
print(f'Mean average recall - {avg_recall:.4f}')
print(f'Mean f1 - {avg_f1:.4f}')
```

```python
classes_inv = {v:k for k,v in classes.items()}
accuracy_by_class = {classes_inv[l]: accuracy_score(all_labels_np[all_labels_np==l], all_preds_np[all_labels_np==l], normalize=True) 
                     for l in np.unique(all_labels_np)}
```

```python
acc_df = pd.DataFrame.from_dict(accuracy_by_class, orient='index', columns=['accuracy'])
acc_df = acc_df.sort_values(by=['accuracy'])
```

```python
acc_df
```

```python
acc_df.plot.bar().set_title("Accuracy by class");
```

```python
print('Average accuracy by class', acc_df.accuracy.mean())
```

```python
cm = confusion_matrix(all_labels, all_preds);
cm
```

```python
# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', 
                          cmap=plt.cm.bwr, print_num=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    if print_num:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

```python
plt.figure(figsize=(8,8))
plot_confusion_matrix(cm, list(classes.keys()), normalize=True)
```

```python
plt.figure(figsize=(7,7))
plot_confusion_matrix(cm, list(classes.keys()), normalize=False)
```

```python

```
