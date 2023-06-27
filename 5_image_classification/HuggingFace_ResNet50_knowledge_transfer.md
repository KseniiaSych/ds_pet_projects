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
import matplotlib.pyplot as plt 

import datasets
from datasets import load_dataset
import evaluate
from transformers import DefaultDataCollator
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import (Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip,
    GaussianBlur, RandomRotation, ToTensor, Normalize)
```

```python
dataroot = '../data/muffin-vs-chihuahua-image-classification'
train_dir = 'train'
test_dir = 'test'

subsets = [train_dir,test_dir ]
class_names = ['chihuahua', 'muffin']
```

```python tags=[]
checkpoint = "microsoft/resnet-50"
```

# View dataset

```python
def print_dataset_len(dataroot, class_name, subsets=[]):
    print(f'{class_name}:')
    lengths = {s:len(os.listdir(os.path.join(dataroot, s, class_name))) for s in subsets}
    total_sum = sum(lengths.values())
    for name, l in lengths.items():
        print(f'{name} - {l} : {round(l/total_sum*100)}%')
```

```python
print("Dataset lenght by class and split:")
for class_name in class_names:
    print_dataset_len(dataroot, class_name, subsets)
```

```python
view_dataset = load_dataset("imagefolder", data_dir=dataroot, split="train").shuffle(seed=42).select(range(9))
```

```python tags=[]
view_dataset
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
    
def plot_batch(dataset = None, image=None, label = None):
    if dataset:
        image = dataset[:9]["image"]
        label = dataset[:9]["label"]
    plt.figure(figsize=(8,8))
    for i in range(1, 10):
        plt.subplot(3,3,i)
        plt.tight_layout()
        plt.imshow(image[i-1])
        plt.xlabel(imglabel(label[i-1]), fontsize=10)
```

```python
plot_batch(dataset = view_dataset)
```

# Train model

```python tags=[]
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
```

```python tags=[]
train_transform = Compose(
    [
            Resize(size),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            RandomRotation(degrees=(30, 70)),
            ToTensor(),
            normalize
    ]
)

test_transform = Compose([
            Resize(size),
            ToTensor(),
            normalize
        ])

def preprocess_train(examples):
    examples["pixel_values"] = [train_transform(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def preprocess_test(examples):
    examples["pixel_values"] = [test_transform(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def preprocess_val(examples):
    examples["pixel_values"] = [test_transform(img.convert("RGB")) for img in examples["image"]]
    return examples
```

```python tags=[]
test_tmp = load_dataset("imagefolder", data_dir=dataroot, split="test")
test_splitted = test_tmp.train_test_split(test_size=0.5, stratify_by_column = "label", shuffle = True )

dataset = datasets.DatasetDict({
    'train': load_dataset("imagefolder", data_dir=dataroot, split="train"),
    'test': test_splitted['test'],
    'valid': test_splitted['train']})
```

```python tags=[]
dataset["train"].set_transform(preprocess_train)
dataset["test"].set_transform(preprocess_test)
```

<!-- #region tags=[] -->
## Model
<!-- #endregion -->

```python tags=[]
class TransferResNet(nn.Module):
    def __init__(self, num_labels): 
        super(TransferResNet,self).__init__() 
        self.model = AutoModelForImageClassification.from_pretrained(checkpoint)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(nn.Linear(1000, 8),
                        nn.ReLU(),  
                        nn.Linear(8, num_labels),
                        nn.Sigmoid())
        
    def loss(self, logits, labels):
        return F.binary_cross_entropy(logits, labels.float())

    def forward(self, pixel_values=None, labels=None):
        representation = self.model(pixel_values, output_hidden_states = True)
        output = self.classifier(representation.logits).flatten()
        if labels is not None:
            loss = self.loss(output, labels)
            return (loss, output)
        return output
```

<!-- #region tags=[] -->
## Train
<!-- #endregion -->

```python tags=[]
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions)
    return accuracy.compute(predictions=predictions, references=labels)
```

```python
model = TransferResNet(1)
data_collator = DefaultDataCollator()
```

```python
training_args = TrainingArguments(
    output_dir="muffin-or-dog",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to = "none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
```

```python tags=[]
trainer.train()
```

```python tags=[]
trainer.evaluate(dataset["valid"])
```

# Visualize results

```python tags=[]
predict_partition = dataset["valid"].shuffle(seed=42).select(range(9))
prepared_predict = predict_partition.map(preprocess_val, remove_columns=["image"], batched=True)
```

```python tags=[]
predictions = trainer.predict(test_dataset = prepared_predict).predictions
```

```python
out = (predictions > 0.5)
plot_batch(image = predict_partition["image"], label = out)
```

```python

```
