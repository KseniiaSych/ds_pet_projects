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
import json, os, time, re, unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re

import datasets
from datasets import load_dataset, DatasetDict

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import langid
from peft import LoraConfig, TaskType, get_peft_model

import ipywidgets as widgets

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
```

# Utils

```python
def save_to_file(filename, data, top_lvl = "data"):
    if top_lvl:
        json_object = json.dumps({top_lvl:data}, indent=4)
    else:
        json_object = json.dumps(data, indent=4)
    with open(os.path.join(path_to_files, filename), "w") as outfile:
        outfile.write(json_object)    

def read_from_file(filename, top_lvl = "data"):
    with open(os.path.join(path_to_files, filename), 'r') as infile:
        d = json.load(infile)
        return d[top_lvl] if top_lvl else d
```

# Preprocess the data

```python
labels_file = "participants.json"
message_files = ["message_2.json", "message_3.json","message_4.json"]
path_to_files ="../data/facebook_messages/"
full_message_dataset = "message_dataset.json"
language_distribution_file = "lang_distribution.json"
distinct_eng_words = "eng_words_distinct.json"
STOPWORDS = set(map(str.strip, open(os.path.join(path_to_files, 'stopwords.txt')).readlines()))
english_words_freq_path = os.path.join(path_to_files, 'unigram_freq.csv')
```

## Define labels

```python
label2id = {}
with open(os.path.join(path_to_files,labels_file)) as file:
    data = json.load(file)["participants"]
    label2id = {participant["name"]:indx for participant, indx in zip(data, range(len(data)))}
id2label = {v:k for k,v in label2id.items()}
print("Found participants and assigned labels:", id2label)
```

## Preprocess files with messages

```python
skip_prefixes = ["cost=", "Obj =", "U = ", "you waved at", ]

def custom_process_message(text):
    for prefix in skip_prefixes:
        if text.startswith(prefix):
            return
    text = re.sub(r'http\S+', '', text)
    text = text.encode('latin1').decode()
    return text
        

def preprocess_message_file(message_path):
    messages = []
    with open(message_path, encoding='utf-8', errors='ignore') as file:
        data = json.load(file)["messages"]
        for message in data:
            if "content" in message:
                if message_processed:=custom_process_message(message["content"]):
                    messages.append({
                        "label": label2id[message["sender_name"]],
                        "text": message_processed
                    })
    return messages
```

```python
result_dataset = []
for message_file in message_files:
    result_dataset.extend(preprocess_message_file(os.path.join(path_to_files, message_file)))
```

## Save dataset to file

```python
save_to_file(full_message_dataset, result_dataset)
```

```python
result_dataset = read_from_file(full_message_dataset)
```

# Analize dataset


## Label distribution

```python
lbl_dist = Counter(d['label'] for d in result_dataset)
g_labels = list(id2label.values())
g_values = list(lbl_dist.values())
```

```python
sns.barplot(x=g_values, hue=g_labels, orient='y').set(title='Distribution of texts by sender')
```

### Distribution by language

```python
distinct_words =  Counter()

def add_to_distinct(text):
    words = [w.strip() for w in re.split(r'(\W+)',  text)]
    filtered_words = [w for w in words if len(w)>1]

    if len(filtered_words)<=1:
        return
    words_counter = Counter(filtered_words)
    distinct_words.update(words_counter)
```

```python
lang_distribution = {}
for label in id2label.keys():
    languages_list = []
    for text_mes in result_dataset:
        if text_mes['label']==label:
            l = langid.classify(text_mes['text'])[0]
            languages_list.append(l)
            if l=='en':
                add_to_distinct(text_mes['text'])
        
    lang_distribution[label]=dict(Counter(languages_list).most_common(5))
```

```python
save_to_file(language_distribution_file, lang_distribution)
save_to_file(distinct_eng_words, distinct_words)
```

```python
lang_distribution = read_from_file(language_distribution_file)
distribution_frame = pd.DataFrame.from_dict(lang_distribution).reset_index(drop=False)
distribution_frame = pd.melt(distribution_frame, id_vars="index", var_name="Label", value_name="Num").rename(columns={"index": "Lang"})
distribution_frame=distribution_frame.dropna()
distribution_frame["Label"] = pd.to_numeric(distribution_frame["Label"]).replace(id2label)

```

```python
distribution_frame["Percentage_By_Person"] = distribution_frame["Num"]/distribution_frame.groupby('Label').transform('sum')["Num"]
distribution_frame["Percentage_By_Lang"] = distribution_frame["Num"]/distribution_frame.groupby('Lang').transform('sum')["Num"]
```

```python
distribution_frame
```

```python
sns.barplot(data = distribution_frame, x ="Lang", y="Percentage_By_Lang", hue="Label").set(title='Percentage for person within language');
```

```python
sns.barplot(data = distribution_frame, x ="Label", y="Percentage_By_Person", hue="Lang").set(title='Percentage for language by person');
```

## Frequencies cloud

```python
STOPWORDS = set(map(str.strip, open(os.path.join(path_to_files, 'stopwords.txt')).readlines()))
filtered_frequencies = {k:v for k,v in distinct_words.items() if k not in STOPWORDS}

wordcloud = WordCloud(max_words = 200, background_color = 'white', relative_scaling=True).generate_from_frequencies(filtered_frequencies)

plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud by frequency")
plt.axis("off")
plt.show()
```

```python
english_words_frequencies = pd.read_csv(english_words_freq_path, index_col=0).to_dict()['count']
```

```python
frequencies_compared_to_eng = {k:(v/english_words_frequencies[k]) for k, v in filtered_frequencies.items() if k in english_words_frequencies}
```

```python
wordcloud = WordCloud(max_words = 200, background_color = 'white', relative_scaling=True).generate_from_frequencies(frequencies_compared_to_eng)

plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud by frequency compared to overall frequency")
plt.axis("off")
plt.show()
```

# Load dataset and split

```python
all_messages_dataset = load_dataset("json", data_files=os.path.join(path_to_files,full_message_dataset), field="data")
```

```python
train_testvalid = all_messages_dataset["train"].shuffle().train_test_split(test_size=0.2)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
messages_dataset = DatasetDict({
    "train": train_testvalid["train"],
    "test": test_valid["test"],
    "valid": test_valid["train"]})
```

```python
messages_dataset
```

# Encode text

```python
lang_model_name = "distilbert-base-multilingual-cased"
lang_tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
```

```python
lang_encoded_dataset = messages_dataset.map(lambda examples:lang_tokenizer(examples['text'], truncation=True, 
                                                                 padding='max_length', max_length = 512),
                                                                 batched=True)
```

```python
base_model_name = "distilbert-base-uncased"
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
```

```python
base_encoded_dataset = messages_dataset.map(lambda examples:base_tokenizer(examples['text'], truncation=True, 
                                                                 padding='max_length', max_length = 512),
                                                                 batched=True)
```

## Visualize tokens

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
message_to_vis = "cześć, jak tam?" #base_encoded_dataset["train"][12]["text"]
print(f" Message: {message_to_vis}")
print("Distilbert-base-uncased")
show_tokens(message_to_vis, base_tokenizer)
print("\n"+"_"*20)
print("Distilbert-base-multilingual-cased")
show_tokens(message_to_vis, lang_tokenizer)
```

# Model

```python
def prepare_model(model_name, tokenizer):
    num_labels = len(id2label.keys()) 
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                                id2label=id2label, label2id=label2id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   
    return model, data_collator

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)
```

```python
def prepare_trainer(model, tokenizer, data_collator, encoded_dataset, epochs, output_dir):
    batch_size = 16
    metric_name = "accuracy"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy = "steps",
        eval_steps = 500,
        save_strategy = "steps",
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_steps=100,
        push_to_hub=False,
        logging_strategy = "steps",
        report_to = "mlflow"
    )
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
    )
    return trainer
```

## Model bert base

```python
bert_base, data_collator = prepare_model(base_model_name, base_tokenizer)
bert_base_trainer = prepare_trainer(bert_base, base_tokenizer, data_collator, base_encoded_dataset, epochs=3, output_dir="base_bert_training")
```

```python
bert_base_trainer.train()
```

```python
bert_base_trainer.save_model("./base_text_classification/")
```

## Model bert multilingual

```python
lang_bert_base, lang_data_collator = prepare_model(lang_model_name, lang_tokenizer)
lang_bert_base_trainer = prepare_trainer(lang_bert_base, lang_tokenizer, lang_data_collator, lang_encoded_dataset, epochs=3, output_dir ="lang_bert_training")
```

```python
lang_bert_base_trainer.train()
```

```python
lang_bert_base_trainer.save_model("./lang_text_classification/")
```

# Infer

```python
model_path = "./lang_text_classification/" #"./base_text_classification/"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

```python
dataset = lang_encoded_dataset # base_encoded_dataset
inputs = dataset['valid'].shuffle()[:20]
```

```python
def model_guess(inputs):
    with torch.no_grad():
        logits = model(input_ids = torch.Tensor(inputs['input_ids']).to(torch.int64),
                   attention_mask = torch.Tensor(inputs['attention_mask']).to(torch.int64)).logits
    predicted_class_ids = logits.argmax(dim=1)
    predicted_list = [p.item() for p in predicted_class_ids]
    results = sum([predicted_list[i] == inputs['label'][i] for i in range(len(predicted_list))])
    return results, predicted_list
```

```python
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
    messages.update(dataset['valid'].shuffle()[:num_messages_to_guess])
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

```python
def on_start(*arg):
    start_game()
    print("here")
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

```python
for i in range(num_messages_to_guess):
    print(f"{messages['text'][i]} ({model.config.id2label[messages['label'][i]]})")
```

# Show model's results

```python
for i, answer_label in enumerate(model_guessed_labels):
    answer_correct = answer_label == messages['label'][i]
    res = '\x1b[32m+\x1b[0m' if answer_correct else '\x1b[31m-\x1b[0m'
    print(f"{res} {messages['text'][i]} (guessed - {model.config.id2label[answer_label]})")
```

# Use adapters for fine-tuning

```python
bert_base
```

```python
config = LoraConfig(
 task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=4, 
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
)
bert_lora = get_peft_model(bert_base, config)
bert_lora_trainer = prepare_trainer(bert_lora, base_tokenizer, data_collator, base_encoded_dataset, epochs=3,  output_dir ="lora_bert_training")
```

```python
bert_lora.print_trainable_parameters()
```

```python
bert_lora_trainer.train()
```

```python

```
