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

```python tags=[]
import json
import os
import re
import unicodedata
import numpy as np

import datasets
from datasets import load_dataset, DatasetDict

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

import evaluate

import ipywidgets as widgets
```

<!-- #region tags=[] -->
# Preprocess the data
<!-- #endregion -->

```python tags=[]
labels_file = "participants.json"
message_files = ["message_2.json", "message_3.json","message_4.json"]
path_to_files ="../data/facebook_messages/"
output_file_name = "message_dataset.json"
```

## Define labels

```python tags=[]
label2id = {}
with open(os.path.join(path_to_files,labels_file)) as file:
    data = json.load(file)["participants"]
    label2id = {participant["name"]:indx for participant, indx in zip(data, range(len(data)))}
id2label = {v:k for k,v in label2id.items()}
print("Found participants and assigned labels:", id2label)
```

## Preprocess files with messages

```python tags=[]
tmp_stats = set()
skip_prefixes = ["cost=", "Obj =", "U = ", "you waved at", ]
punc = '''()-[]{};:"\,<>/@#$%^&*_~'''

def process_message(text):
    for prefix in skip_prefixes:
        if text.startswith(prefix):
            return
    text = re.sub(r'http\S+', '', text)
    for punctuation in punc:
        text = text.replace(punctuation, '')
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')
    words = text.split()
    tmp_stats.update(words)
    if len(words)<=1:
        return
    return text.lower()
        

def preprocess_message_file(message_path):
    messages = []
    with open(message_path) as file:
        data = json.load(file)["messages"]
        for message in data:
            if "content" in message:
                if message_processed:=process_message(message["content"]):
                     messages.append({
                        "label": label2id[message["sender_name"]],
                        "text": message_processed
                    })
    return messages
```

```python tags=[]
result_dataset = []
for message_file in message_files:
    result_dataset.extend(preprocess_message_file(os.path.join(path_to_files, message_file)))

json_object = json.dumps({"data":result_dataset}, indent=4)
with open(os.path.join(path_to_files,output_file_name), "w") as outfile:
    outfile.write(json_object)    
```

<!-- #region tags=[] -->
## Load dataset and split
<!-- #endregion -->

```python tags=[]
all_messages_dataset = load_dataset("json", data_files=os.path.join(path_to_files,output_file_name), field="data")
```

```python tags=[]
train_testvalid = all_messages_dataset["train"].shuffle().train_test_split(test_size=0.2)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
messages_dataset = DatasetDict({
    "train": train_testvalid["train"],
    "test": test_valid["test"],
    "valid": test_valid["train"]})
```

```python tags=[]
messages_dataset
```

# Encode text

```python tags=[]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

```python tags=[]
encoded_dataset = messages_dataset.map(lambda examples:tokenizer(examples['text'], truncation=True, 
                                                                 padding='max_length', max_length = 512),
                                                                 batched=True)
```

<!-- #region tags=[] -->
# Model
<!-- #endregion -->

```python tags=[]
num_labels = len(id2label.keys()) 
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels,
                                                            id2label=id2label, label2id=label2id)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

```python tags=[]
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
```

```python tags=[]
batch_size = 16
epochs = 20
metric_name = "accuracy"

training_args = TrainingArguments(
    output_dir="messages_classified",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_steps=10,
    push_to_hub=False,
    report_to = "none"
)
```

```python tags=[]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
```

```python tags=[]
trainer.train()
```

# Save model

```python tags=[]
model_path = "./text_classification/"
```

```python tags=[]
trainer.save_model(model_path)
```

# Infer

```python tags=[]
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

```python tags=[]
inputs = encoded_dataset['valid'].shuffle()[:20]
```

```python tags=[]
def model_guess(inputs):
    with torch.no_grad():
        logits = model(input_ids = torch.Tensor(inputs['input_ids']).to(torch.int64),
                   attention_mask = torch.Tensor(inputs['attention_mask']).to(torch.int64)).logits
    predicted_class_ids = logits.argmax(dim=1)
    predicted_list = [p.item() for p in predicted_class_ids]
    results = sum([predicted_list[i] == inputs['label'][i] for i in range(len(predicted_list))])
    return results, predicted_list
```

```python tags=[]
class Counter():
    def __init__(self):
        self.value = 0
    
    def inc(self):
        self.value+=1
    
    def reset(self):
        self.value = 0
```

```python
num_messages_to_guess = 10
messages = {}
counter = Counter()
guessed = Counter() 

model_guessed_labels = []


def show_messsage():
    message.value = messages['text'][counter.value]

    
def start_game():
    model_guessed_labels.clear()
    messages.clear()
    messages.update(encoded_dataset['valid'].shuffle()[:num_messages_to_guess])
    counter.reset()
    guessed.reset()
    show_messsage()
    

def step():
    answer = model.config.label2id[pick_sender.value]
    correct = messages['label'][counter.value]
    if answer == correct:
        guessed.inc()
    counter.inc()
    if counter.value==num_messages_to_guess:
        return 'end'
    else:
        show_messsage()

        
def show_resuts():
    model_guessed, model_results = model_guess(messages)
    model_guessed_labels.extend(model_results)
    return f"You've guessed correctly {guessed.value}/{num_messages_to_guess}. Model guessed correctly {model_guessed}/{num_messages_to_guess}"
```

```python tags=[]
def on_start(*arg):
    start_game()
    start_button.layout.visibility = 'hidden'
    result.layout.visibility = 'hidden'
    message.layout.visibility = 'visible'
    pick_sender.layout.visibility = 'visible'
    submit_answer.layout.visibility = 'visible'


def on_submit(*arg):
    if step()=='end':
        start_button.layout.visibility = 'visible'
        result.layout.visibility = 'visible'
        message.layout.visibility = 'hidden'
        pick_sender.layout.visibility = 'hidden'
        submit_answer.layout.visibility = 'hidden'
        result.value = 'Loading...'
        result.value = show_resuts()
        
    
message = widgets.Textarea(
    value='',
    disabled=False,
    layout = widgets.Layout(width='800px')
)
message.layout.visibility = 'hidden'

result = widgets.Textarea(
    value='',
    disabled=False,
    layout = widgets.Layout(width='500px')
)
result.layout.visibility = 'hidden'

pick_sender = widgets.Select(
    options=model.config.label2id.keys(),
    disabled=False
)
pick_sender.layout.visibility = 'hidden'

submit_answer = widgets.Button(
    description='Submit',
    disabled=False,
    button_style=''
)
submit_answer.on_click(on_submit)
submit_answer.layout.visibility = 'hidden'
        
start_button = widgets.Button(
    description='Start_game',
    disabled=False,
    button_style=''
)
start_button.on_click(on_start)

display(start_button)
display(message)
display(result)
display(pick_sender)
display(submit_answer)
```

# Actual answers

```python tags=[]
for i in range(num_messages_to_guess):
    print(f"{messages['text'][i]} ({model.config.id2label[messages['label'][i]]})")
```

# Show model's results

```python tags=[]
for i, answer_label in enumerate(model_guessed_labels):
    answer_correct = answer_label == messages['label'][i]
    res = '\x1b[32m+\x1b[0m' if answer_correct else '\x1b[31m-\x1b[0m'
    print(f"{res} {messages['text'][i]} (guessed - {model.config.id2label[answer_label]})")
```

```python

```
