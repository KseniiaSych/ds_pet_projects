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

import torch

import datasets
from datasets import load_dataset, DatasetDict

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import pipeline
```

# Data preparation

```python
chat_datapath ="../data/facebook_messages/"
chat_file = "message_dataset.json"
model_output = "./gpt_trained"
```

```python
def preprocess_message_file(datapath, filepath):    
    with open(os.path.join(datapath,filepath)) as file:
        data = json.load(file)
        messages = []
        for message in data["data"]:
            if message.get("label", "")==0 and message["text"]:
                text = message["text"]
                text = re.sub(r'http\S+', '', text)
                messages.append({ "text": text })
        return messages

def save_to_file(filename, data):
    json_object = json.dumps(data, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_object)   
```

```python
texts = preprocess_message_file(chat_datapath, chat_file)
prepared_chat = {"data": texts}
processed_chat = f"processed_for_generating_{chat_file}"
save_to_file(os.path.join(chat_datapath, processed_chat), prepared_chat)
```

# Create dataset

```python
def prepare_dataset(datapath, chat_path, tokenizer):
    all_messages_dataset = load_dataset("json", data_files=os.path.join(datapath,chat_path), field="data")
    train_testvalid = all_messages_dataset["train"].shuffle().train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    messages_dataset = DatasetDict({
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]})
    print("Dataset ", chat_path)
    print(messages_dataset)
        
    messages_dataset = messages_dataset.map(lambda element:tokenizer(element['text'], truncation=True, padding='max_length', max_length=512))

    return messages_dataset
```

```python
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
```

```python
messages_dataset = prepare_dataset(chat_datapath, processed_chat, tokenizer)
```

```python
messages_dataset["train"]
```

## Visualize tokenisation

```python
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence, tokenizer):
    """ Show the tokens each separated by a different color """

    # Load the tokenizer and tokenize the input
    token_ids = tokenizer(sentence).input_ids

    # Extract vocabulary length
    print(f"Vocab length: {len(tokenizer)}")

    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )
```

```python
for message_to_vis in messages_dataset["train"][:10]["text"]:
    print(f" Message: {message_to_vis}")
    show_tokens(message_to_vis, tokenizer)
    print("\n"+"_"*20)
```

# Load model

```python
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", device_map="auto")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

# Fine-tune

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

```python
batch_size=8
epochs = 12

training_args = TrainingArguments(
    output_dir="./gpt_train",
    eval_strategy = "steps",
    eval_steps = 500,
    save_strategy = "steps",
    save_steps=500,
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    push_to_hub=False,
    logging_strategy = "steps",
    report_to = "mlflow"
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=messages_dataset["train"],
    eval_dataset=messages_dataset["test"]
)
```

```python
#trainer.train(resume_from_checkpoint="./gpt_train/checkpoint-11620")
trainer.train()
```

```python
model.save_pretrained(model_output)
tokenizer.save_pretrained(model_output)
```

# Infer

```python
checkpoint_path=model_output
origin_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", device_map="auto")
model = GPT2LMHeadModel.from_pretrained(checkpoint_path, device_map="auto")
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
```

```python
pipeline_origin = pipeline(task="text-generation", model=origin_model, tokenizer=tokenizer)
pipeline_chat = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
```

```python
stop_strings = [".","!", "?", "XD"]
for pipeine, name in zip([pipeline_origin, pipeline_chat], ["Origin", "Fine_tuned"]):
    print(f"{name} model:")
    for i in range(3):
        print( pipeline_chat("Today I had", num_return_sequences=1, return_full_text=True,  stop_strings=stop_strings,  tokenizer=tokenizer)[0]['generated_text'])
        print("-"*20)
```

```python

```
