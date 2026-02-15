---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
import json
import re
import numpy as np

import datasets
from datasets import load_dataset, DatasetDict
import torch

from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
import evaluate

import adapters
from adapters import LoRAConfig

from adapters import AdapterTrainer
from transformers import DataCollatorForLanguageModeling

import ipywidgets as widgets
```

# Preprocess data

```python
telegram_datapath ="../data/telegram_chats/"
available_chats = [file for file in os.listdir(telegram_datapath) if file.endswith(".json") and not file.startswith("processed_")]
```

```python
print("Found chats :", available_chats)
```

```python
skip_prefixes = ["cost=", "Obj =", "U = ", "you waved at", ]

def custom_process_message(text):
    for prefix in skip_prefixes:
        if text.startswith(prefix):
            return
    text = re.sub(r'http\S+', '', text)
    return text
        
def preprocess_message_file(datapath, filepath):
    
    with open(os.path.join(datapath,filepath)) as file:
        data = json.load(file)
        labels = data["labels"]
        messages = []
        for message in data["messages"]:
            if len(message["text_entities"])>0:
                text = ''.join([m["text"] for m in message["text_entities"]])
                if message_processed:=custom_process_message(text):
                    messages.append({
                                "label": labels[message["from"]],
                                "text": message_processed
                            })
        return labels, messages

def save_to_file(filename, data, top_lvl = "data"):
    if top_lvl:
        json_object = json.dumps({top_lvl:data}, indent=4)
    else:
        json_object = json.dumps(data, indent=4)
    with open(os.path.join(path_to_files, filename), "w") as outfile:
        outfile.write(json_object)    
```

```python
for chat_file in available_chats:
    labels, texts = preprocess_message_file(telegram_datapath, chat_file)
    prepared_chat = {"data": texts, "labels":labels}
    with open(os.path.join(telegram_datapath, f"processed_{chat_file}"), "w") as outfile:
        json_object = json.dumps(prepared_chat, indent=4)
        outfile.write(json_object)    
```

# Common for train and infer

```python
model_name = "distilbert-base-multilingual-cased"
def compose_adapter_outpath(file_path):
     return file_path[10:-5]+"_classification"

adapter_name_train_name = "chat_adapter"

def prepare_dataset(datapath, chat_path, model_name):
    all_messages_dataset = load_dataset("json", data_files=os.path.join(datapath,chat_path), field="data")
    train_testvalid = all_messages_dataset["train"].shuffle().train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    messages_dataset = DatasetDict({
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]})
    print("Dataset ", chat_path)
    print(messages_dataset)

    with open(os.path.join(datapath, chat_path), 'r') as chat_file:
        labels = json.load(chat_file)["labels"]
        id2label = {v:k for k, v in labels.items()}
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_input_names=['input_ids'])
    messages_dataset = messages_dataset.map(lambda element:tokenizer(element['text'], truncation=True, 
                                                                 padding='max_length', max_length = 512,
                                                                 add_special_tokens=False))

    return messages_dataset, labels, id2label, tokenizer

```

# Train Bert with adapters

```python
available_prepared_chats = [file for file in os.listdir(telegram_datapath) if file.startswith("processed_")]
```

```python
print("Available prepared chats ", available_prepared_chats)
```

```python

def prepare_model(model_name, id2label, label2id):
    num_labels = len(id2label.keys()) 
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                                id2label=id2label, label2id=label2id)
    adapters.init(model)
    config = LoRAConfig(selfattn_lora=True, intermediate_lora=True, output_lora=True, attn_matrices=["q", "k", "v"], alpha=16, r=64, dropout=0.1)
    model.add_adapter(adapter_name_train_name, config=config)
    model.train_adapter(adapter_name_train_name)
    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)


def prepare_trainer(model, tokenizer, encoded_dataset, epochs, output_dir):
    batch_size = 32
    metric_name = "accuracy"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy = "steps",
        eval_steps = 200,
        save_strategy = "steps",
        save_steps=400,
        
        lr_scheduler_type="constant",
        learning_rate=0.0002,
        group_by_length=True,
        warmup_ratio=0.03,
        
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_steps=100,
        push_to_hub=False,
        logging_strategy = "steps",
        report_to = "mlflow"
    )
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
        )
    return trainer
```

```python
def train_adapter(chat_file_path):
    messages_dataset, labels, id_labels, tokenizer = prepare_dataset(telegram_datapath, chat_file_path, model_name)
    model = prepare_model(model_name, id_labels, labels)
    trainer = prepare_trainer(model, tokenizer, messages_dataset, epochs=5, output_dir = compose_adapter_outpath(chat_file_path))
    trainer.train()
```

```python
for available_chat in available_prepared_chats:
    train_adapter(available_chat)
```

# Load adapters

```python
model = DistilBertForSequenceClassification.from_pretrained(model_name) 
adapters.init(model)
```

```python
def resolve_adapter_path(path, adapter_name):
    checkpoints = os.listdir(path)
    return os.path.join(path, sorted(checkpoints)[-1], adapter_name)
```

```python
available_prepared_chats = [file for file in os.listdir(telegram_datapath) if file.startswith("processed_")]
chat_datasets = {}
for chat_file_path in available_prepared_chats:
    messages_dataset, labels, id_labels, _ = prepare_dataset(telegram_datapath, chat_file_path, model_name)
    chat_datasets[chat_file_path[10:-5]] = {"dataset": messages_dataset, "labels": labels, "id2labels": id_labels}
```

```python
adapters_for_chats = [compose_adapter_outpath(chat_file_path) for chat_file_path in available_prepared_chats]
available_adapters = [adapter for adapter in adapters_for_chats if os.path.exists(adapter)]

loaded_adapters = [model.load_adapter(resolve_adapter_path(adapter, adapter_name_train_name), load_as= adapter[:-15]) for  adapter in available_adapters]

print("Loaded adapters: ", loaded_adapters)
```

```python
def model_guess(inputs):
    with torch.no_grad():
       logits = model(input_ids = torch.Tensor(inputs['input_ids']).to(torch.int64)).logits
    predicted_class_ids = logits.argmax(dim=1)
    predicted_list = [p.item() for p in predicted_class_ids]
    results = sum([predicted_list[i] == inputs['label'][i] for i in range(len(predicted_list))])
    return results, predicted_list

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
current_chat = ""

model_guessed_labels = []

def show_messsage():
    message.value = messages['text'][counter.value]

def set_up_chat(adapter_name):
    current_chat = adapter_name
    model.set_active_adapters(adapter_name)
    model.label2id = chat_datasets[adapter_name]["labels"]
    model.id2label = chat_datasets[adapter_name]["id2labels"]
    dataset = chat_datasets[adapter_name]["dataset"]
    inputs = dataset['valid'].shuffle()[:num_messages_to_guess]
    return inputs
    
    
def start_game(inputs):
    model_guessed_labels.clear()
    messages.clear()
    messages.update(inputs)
    counter.reset()
    guessed.reset()
    show_messsage()
    

def step():
    answer = model.label2id[pick_sender.value]
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

```python
def on_start(*arg):
    chat_name = chat_options.value
    if chat_name not in loaded_adapters:
        print("Choose chat")
        return
    inputs = set_up_chat(chat_name)
    pick_sender.options = list(model.label2id.keys())
    
    start_game(inputs)
    start_button.layout.visibility = 'hidden'
    result.layout.visibility = 'hidden'
    message.layout.visibility = 'visible'
    pick_sender.layout.visibility = 'visible'
    submit_answer.layout.visibility = 'visible'
    actual.layout.visibility = 'hidden'

def combine_actual_labels():
    res = []
    for i in range(num_messages_to_guess):
        res.append(f"{messages['text'][i]} ({model.id2label[messages['label'][i]]})")
    return '\n'.join(res)

def on_submit(*arg):
    if step()=='end':
        start_button.layout.visibility = 'visible'
        result.layout.visibility = 'visible'
        message.layout.visibility = 'hidden'
        pick_sender.layout.visibility = 'hidden'
        submit_answer.layout.visibility = 'hidden'
        result.value = 'Loading...'
        result.value = show_resuts()

        actual.layout.visibility = 'visible'
        actual.value = combine_actual_labels()
        
        

def on_chat_changed(*args):
    if chat_options.value == current_chat:
        return
    start_button.layout.visibility = 'visible'
    result.layout.visibility = 'visible'
    message.layout.visibility = 'hidden'
    pick_sender.layout.visibility = 'hidden'
    submit_answer.layout.visibility = 'hidden'
    result.layout.visibility = 'hidden'
    actual.layout.visibility = 'hidden'
    
message = widgets.Textarea(
    value='',
    disabled=False,
    layout = widgets.Layout(width='800px',  height="100px")
)
message.layout.visibility = 'hidden'

result = widgets.Textarea(
    value='',
    disabled=False,
    layout = widgets.Layout(width='500px',  height="50px")
)
result.layout.visibility = 'hidden'

pick_sender = widgets.Select(
    options=[],
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

chat_options = widgets.Dropdown(
    options=loaded_adapters,
    description='Choose chat:',
    disabled=False,
)

chat_options.observe(on_chat_changed)

actual = widgets.Textarea(
    value='',
    disabled=False,
    layout = widgets.Layout(width='800px', height=f'{num_messages_to_guess*25}px')
)

actual.layout.visibility = 'hidden'

display(chat_options)
display(start_button)
display(message)
display(result)
display(pick_sender)
display(submit_answer)
display(actual)
```

## Memory usage comparision

```python
model_bert = DistilBertForSequenceClassification.from_pretrained(model_name) 
```

```python
print(torch.cuda.memory_summary())
```

```python

```
