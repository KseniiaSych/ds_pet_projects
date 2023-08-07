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
import numpy as np

import datasets
from datasets import load_dataset
from transformers import DefaultDataCollator
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate

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
subsets = [train_dir, test_dir ]

label2id = {"chihuahua": 0, "muffin": 1}
id2label = {v: k for k, v in label2id.items()}

class_names = list(label2id.keys())
```

```python tags=[]
checkpoint = "microsoft/resnet-50"
batch_size = 16
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
            print(f'{name} - {l} :{l/total_sum*100:.2f}%')
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
    
def plot_batch(dataset = None, image=None, label = None):
    if dataset:
        image = dataset[:9]["image"]
        label = dataset[:9]["label"]
    plt.figure(figsize=(8,8))
    for i in range(1, 10):
        plt.subplot(3,3,i)
        plt.tight_layout()
        plt.imshow(image[i-1])
        plt.xlabel(id2label[label[i-1]], fontsize=10)
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

test_transform = Compose([Resize(size), ToTensor(), normalize])

def preprocess(examples, transform):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

preprocess_train = lambda examples: preprocess(examples, train_transform)
preprocess_test = lambda examples: preprocess(examples, test_transform)
```

```python tags=[]
train_tmp = load_dataset("imagefolder", data_dir=dataroot, split="train")
train_split = train_tmp.train_test_split(test_size=0.8, stratify_by_column = "label", shuffle = True)

dataset = datasets.DatasetDict({
    'train': train_split['test'],
    'test': load_dataset("imagefolder", data_dir=dataroot, split="test"),
    'valid': train_split['train']})
```

```python tags=[]
dataset["train"].set_transform(preprocess_train)
dataset["test"].set_transform(preprocess_test)
dataset["valid"].set_transform(preprocess_test)
```

<!-- #region tags=[] -->
## Train
<!-- #endregion -->

```python
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True)
```

```python tags=[]
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

```python
training_args = TrainingArguments(
    output_dir="muffin-or-dog",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to = "none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
```

```python tags=[]
train_results = trainer.train()
trainer.log_metrics("train", train_results.metrics)
```

```python
test_metrics = trainer.evaluate()
trainer.log_metrics("eval", test_metrics)
```

```python tags=[]
trainer.evaluate(dataset["valid"])
```

# Visualize results

```python tags=[]
prepared_predict = dataset["valid"].shuffle(seed=42).select(range(9))
```

```python tags=[]
predictions = trainer.predict(test_dataset = prepared_predict).predictions[:,0]
```

```python
out = np.zeros_like(predictions)
threshold = 0.5
out[predictions > threshold] = 1

images = [predict['image'] for predict in prepared_predict]
plot_batch(image = images, label = np.asarray(out))
```

```python

```
