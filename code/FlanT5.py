#!/usr/bin/env python
# coding: utf-8

# # Flan T5 codice iniziale

# In[ ]:


get_ipython().system('pip install accelerate -U')
get_ipython().system('pip install transformers[torch] -U')
get_ipython().system('pip install scikit-learn')


# In[ ]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Definizione del percorso dei dati
data_path = '/content/drive/My Drive/esperimento'

# Funzione per caricare i dati
def load_data(data_path):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label_folder, label_index in label_dict.items():
        folder_path = os.path.join(data_path, label_folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(label_index)
                titles.append(filename)
    return texts, labels, titles

# Caricamento dei dati
texts, labels, titles = load_data(data_path)

# Tokenizzazione dei dati
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Creazione della classe Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Configurazione del Trainer
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

def compute_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions

    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}

# Cross-Validation
skf = StratifiedKFold(n_splits=5)
for fold, (train_index, test_index) in enumerate(skf.split(texts, labels)):
    print(f"Fold {fold + 1}")

    train_texts = [texts[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    train_labels = [labels[i] for i in train_index]
    test_labels = [labels[i] for i in test_index]
    train_titles = [titles[i] for i in train_index]
    test_titles = [titles[i] for i in test_index]

    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-small", num_labels=len(set(labels)))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = get_training_args(output_dir=f'./results_fold_{fold + 1}')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Avvio del training e della valutazione
    trainer.train()
    eval_results = trainer.evaluate()

    # Valutazione manuale sui dati di test
    predictions = trainer.predict(test_dataset)
    eval_results = compute_metrics(predictions)
    predicted_labels = np.argmax(predictions.predictions[0], axis=1)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

    # Stampa del classification report formattato
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        class_report = report[class_name]
        print(f"{class_name}\t{class_report['precision']:.2f}\t\t{class_report['recall']:.2f}\t{class_report['f1-score']:.2f}\t\t{class_report['support']}")

    print("\n\t\t\tSupport")
    print(f"Accuracy\t\t{report['accuracy']:.2f}\t{len(test_labels)}")
    print(f"Macro avg\t{report['macro avg']['precision']:.2f}\t\t{report['macro avg']['recall']:.2f}\t{report['macro avg']['f1-score']:.2f}\t\t{len(test_labels)}")
    print(f"Weighted avg\t{report['weighted avg']['precision']:.2f}\t\t{report['weighted avg']['recall']:.2f}\t{report['weighted avg']['f1-score']:.2f}\t\t{len(test_labels)}")

    # Funzione per la stampa degli errori usando il nome del file come titolo
    def print_errors(titles, test_labels, predicted_labels, target_names, max_errors=10):
        print("\nErrori nei testi di test:")
        error_count = 0
        for i, (title, true_label, predicted_label) in enumerate(zip(titles, test_labels, predicted_labels)):
            if true_label != predicted_label:
                print(f"\nTitolo: {title}")
                print(f"Etichetta reale: {target_names[true_label]}")
                print(f"Etichetta predetta: {target_names[predicted_label]}")
                error_count += 1
                if error_count >= max_errors:
                    break
        if error_count == 0:
            print("Nessun errore trovato.")

    # Utilizza la funzione per la stampa degli errori
    print_errors(test_titles, test_labels, predicted_labels, target_names)


# # Flan T5 italiano

# In[ ]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Definizione del percorso dei dati
data_path = '/content/drive/My Drive/esperimento'

# Funzione per caricare i dati
def load_data(data_path):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label_folder, label_index in label_dict.items():
        folder_path = os.path.join(data_path, label_folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(label_index)
                titles.append(filename)
    return texts, labels, titles

# Caricamento dei dati
texts, labels, titles = load_data(data_path)

# Tokenizzazione dei dati
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Creazione della classe Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Configurazione del Trainer
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

def compute_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions

    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}

# Dizionari per raccogliere i risultati medi
precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

accuracy_per_fold = []
macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

# Cross-Validation usando KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(texts)):
    print(f"Fold {fold + 1}")

    train_texts = [texts[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    train_labels = [labels[i] for i in train_index]
    test_labels = [labels[i] for i in test_index]
    train_titles = [titles[i] for i in train_index]
    test_titles = [titles[i] for i in test_index]

    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-small", num_labels=len(set(labels)))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = get_training_args(output_dir=f'./results_fold_{fold + 1}')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Avvio del training e della valutazione
    trainer.train()
    eval_results = trainer.evaluate()

    # Valutazione manuale sui dati di test
    predictions = trainer.predict(test_dataset)
    eval_results = compute_metrics(predictions)
    predicted_labels = np.argmax(predictions.predictions[0], axis=1)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

    # Stampa del classification report formattato
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        class_report = report[class_name]
        print(f"{class_name}\t{class_report['precision']:.2f}\t\t{class_report['recall']:.2f}\t{class_report['f1-score']:.2f}\t\t{class_report['support']}")

    print("\n\t\t\tSupport")
    print(f"Accuracy\t\t{report['accuracy']:.2f}\t{len(test_labels)}")
    print(f"Macro avg\t{report['macro avg']['precision']:.2f}\t\t{report['macro avg']['recall']:.2f}\t{report['macro avg']['f1-score']:.2f}\t\t{len(test_labels)}")
    print(f"Weighted avg\t{report['weighted avg']['precision']:.2f}\t\t{report['weighted avg']['recall']:.2f}\t{report['weighted avg']['f1-score']:.2f}\t\t{len(test_labels)}")

    # Somma i valori di precision, recall, f1 e support per ogni classe
    for class_name in target_names:
        precision_sum[class_name] += report[class_name]['precision']
        recall_sum[class_name] += report[class_name]['recall']
        f1_sum[class_name] += report[class_name]['f1-score']
        support_sum[class_name] += report[class_name]['support']

    # Somma per macro avg e weighted avg
    macro_avg_precision_sum += report['macro avg']['precision']
    macro_avg_recall_sum += report['macro avg']['recall']
    macro_avg_f1_sum += report['macro avg']['f1-score']

    weighted_avg_precision_sum += report['weighted avg']['precision']
    weighted_avg_recall_sum += report['weighted avg']['recall']
    weighted_avg_f1_sum += report['weighted avg']['f1-score']

    # Aggiungi l'accuratezza alla lista per calcolare la media finale
    accuracy_per_fold.append(report['accuracy'])

# Media dei risultati per ogni classe
print("\nRisultati medi su 5 fold:")
print("\t\tPrecision\tRecall\tF1-score\tSupport")
for class_name in target_names:
    print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

# Media per Macro avg e Weighted avg
print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

# Stampa dell'accuratezza media finale
print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")


# # Flan T5 inglese

# In[ ]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Definizione del percorso dei dati
data_path = '/content/drive/My Drive/Esperimento_eng'

# Funzione per caricare i dati
def load_data(data_path):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label_folder, label_index in label_dict.items():
        folder_path = os.path.join(data_path, label_folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(label_index)
                titles.append(filename)
    return texts, labels, titles

# Caricamento dei dati
texts, labels, titles = load_data(data_path)

# Tokenizzazione dei dati
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Creazione della classe Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Configurazione del Trainer
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

def compute_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions

    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}

# Dizionari per raccogliere i risultati medi
precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

accuracy_per_fold = []
macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

# Cross-Validation usando KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(texts)):
    print(f"Fold {fold + 1}")

    train_texts = [texts[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    train_labels = [labels[i] for i in train_index]
    test_labels = [labels[i] for i in test_index]
    train_titles = [titles[i] for i in train_index]
    test_titles = [titles[i] for i in test_index]

    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-small", num_labels=len(set(labels)))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = get_training_args(output_dir=f'./results_fold_{fold + 1}')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Avvio del training e della valutazione
    trainer.train()
    eval_results = trainer.evaluate()

    # Valutazione manuale sui dati di test
    predictions = trainer.predict(test_dataset)
    eval_results = compute_metrics(predictions)
    predicted_labels = np.argmax(predictions.predictions[0], axis=1)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

    # Stampa del classification report formattato
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        class_report = report[class_name]
        print(f"{class_name}\t{class_report['precision']:.2f}\t\t{class_report['recall']:.2f}\t{class_report['f1-score']:.2f}\t\t{class_report['support']}")

    print("\n\t\t\tSupport")
    print(f"Accuracy\t\t{report['accuracy']:.2f}\t{len(test_labels)}")
    print(f"Macro avg\t{report['macro avg']['precision']:.2f}\t\t{report['macro avg']['recall']:.2f}\t{report['macro avg']['f1-score']:.2f}\t\t{len(test_labels)}")
    print(f"Weighted avg\t{report['weighted avg']['precision']:.2f}\t\t{report['weighted avg']['recall']:.2f}\t{report['weighted avg']['f1-score']:.2f}\t\t{len(test_labels)}")

    # Somma i valori di precision, recall, f1 e support per ogni classe
    for class_name in target_names:
        precision_sum[class_name] += report[class_name]['precision']
        recall_sum[class_name] += report[class_name]['recall']
        f1_sum[class_name] += report[class_name]['f1-score']
        support_sum[class_name] += report[class_name]['support']

    # Somma per macro avg e weighted avg
    macro_avg_precision_sum += report['macro avg']['precision']
    macro_avg_recall_sum += report['macro avg']['recall']
    macro_avg_f1_sum += report['macro avg']['f1-score']

    weighted_avg_precision_sum += report['weighted avg']['precision']
    weighted_avg_recall_sum += report['weighted avg']['recall']
    weighted_avg_f1_sum += report['weighted avg']['f1-score']

    # Aggiungi l'accuratezza alla lista per calcolare la media finale
    accuracy_per_fold.append(report['accuracy'])

# Media dei risultati per ogni classe
print("\nRisultati medi su 5 fold:")
print("\t\tPrecision\tRecall\tF1-score\tSupport")
for class_name in target_names:
    print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

# Media per Macro avg e Weighted avg
print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

# Stampa dell'accuratezza media finale
print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")


# # Flan T5 con badwords italiano

# In[ ]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Scarica risorse di NLTK per la tokenizzazione
nltk.download('punkt')

# Definizione del percorso dei dati
data_path = '/content/drive/My Drive/esperimento'

# Caricamento delle bad words in italiano
bad_words_path = '/content/drive/My Drive/bad_words.txt'

try:
    with open(bad_words_path, 'r', encoding='utf-8') as file:
        bad_words = set(line.strip() for line in file)
except Exception as e:
    print(f"Error loading bad words: {e}")
    bad_words = set()  # Definisci un set vuoto in caso di errore per evitare crash

# Funzione per rilevare bad words
def detect_bad_words(text, bad_words):
    words = set(word_tokenize(text.lower()))  # Tokenizza e abbassa il testo
    return words.intersection(bad_words)

# Funzione per caricare i dati
def load_data(data_path):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label_folder, label_index in label_dict.items():
        folder_path = os.path.join(data_path, label_folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(label_index)
                titles.append(filename)
    return texts, labels, titles

# Caricamento dei dati
texts, labels, titles = load_data(data_path)

# Tokenizzazione dei dati
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Creazione della classe Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Configurazione del Trainer
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

def compute_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions

    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}

# Dizionari per raccogliere i risultati medi
precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

accuracy_per_fold = []
macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

# Lista per memorizzare i testi in cui l'etichetta è cambiata
texts_with_label_changes = []

# Cross-Validation usando KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(texts)):
    print(f"Fold {fold + 1}")

    train_texts = [texts[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    train_labels = [labels[i] for i in train_index]
    test_labels = [labels[i] for i in test_index]
    train_titles = [titles[i] for i in train_index]
    test_titles = [titles[i] for i in test_index]

    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-small", num_labels=len(set(labels)))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = get_training_args(output_dir=f'./results_fold_{fold + 1}')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Avvio del training e della valutazione
    trainer.train()
    eval_results = trainer.evaluate()

    # Valutazione manuale sui dati di test
    predictions = trainer.predict(test_dataset)
    eval_results = compute_metrics(predictions)
    predicted_labels = np.argmax(predictions.predictions[0], axis=1)

    # Aggiungi il controllo delle bad words
    adjusted_labels = []
    detected_bad_words_list = []

    for i, text in enumerate(test_texts):
        detected_bad_words = detect_bad_words(text, bad_words)
        original_label = predicted_labels[i]
        if original_label == 0 and detected_bad_words:
            adjusted_labels.append(2)  # Cambia l'etichetta in "adulti" se ci sono bad words
            detected_bad_words_list.append((test_titles[i], detected_bad_words))

            # Memorizza i testi modificati
            texts_with_label_changes.append({
                "Title": test_titles[i],
                "Original Text": text,
                "Detected Bad Words": list(detected_bad_words),
                "Original Label": original_label,  # Etichetta originale (0 = bambini)
                "New Label": 2  # Nuova etichetta (2 = adulti)
            })
        else:
            adjusted_labels.append(original_label)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(test_labels, adjusted_labels, target_names=target_names, output_dict=True)

    # Somma i valori di precision, recall, f1 e support per ogni classe
    for class_name in target_names:
        precision_sum[class_name] += report[class_name]['precision']
        recall_sum[class_name] += report[class_name]['recall']
        f1_sum[class_name] += report[class_name]['f1-score']
        support_sum[class_name] += report[class_name]['support']

    # Somma per macro avg e weighted avg
    macro_avg_precision_sum += report['macro avg']['precision']
    macro_avg_recall_sum += report['macro avg']['recall']
    macro_avg_f1_sum += report['macro avg']['f1-score']

    weighted_avg_precision_sum += report['weighted avg']['precision']
    weighted_avg_recall_sum += report['weighted avg']['recall']
    weighted_avg_f1_sum += report['weighted avg']['f1-score']

    # Aggiungi l'accuratezza alla lista per calcolare la media finale
    accuracy_per_fold.append(report['accuracy'])

    # Stampa del classification report formattato
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        class_report = report[class_name]
        print(f"{class_name}\t{class_report['precision']:.2f}\t\t{class_report['recall']:.2f}\t{class_report['f1-score']:.2f}\t\t{class_report['support']}")

    print("\n\t\t\tSupport")
    print(f"Accuracy\t\t{report['accuracy']:.2f}\t{len(test_labels)}")
    print(f"Macro avg\t{report['macro avg']['precision']:.2f}\t\t{report['macro avg']['recall']:.2f}\t{report['macro avg']['f1-score']:.2f}\t\t{len(test_labels)}")
    print(f"Weighted avg\t{report['weighted avg']['precision']:.2f}\t\t{report['weighted avg']['recall']:.2f}\t{report['weighted avg']['f1-score']:.2f}\t\t{len(test_labels)}")

    # Stampa i testi in cui sono state rilevate bad words
    if detected_bad_words_list:
        print(f"Bad words rilevate nel fold {fold + 1}:")
        for title, bad_words in detected_bad_words_list:
            print(f"Titolo: {title}, Bad words: {bad_words}")

# Creazione della tabella riassuntiva finale
print("\n\n===== Tabella Riassuntiva dei Risultati Medi su 5 Fold =====")
print("\t\tPrecision\tRecall\tF1-score\tSupport")
for class_name in target_names:
    print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

# Stampa dell'accuratezza media finale
print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")

# Stampa dei testi in cui l'etichetta è stata cambiata a causa delle bad words
if texts_with_label_changes:
    print(f"\nTesti con cambiamento di etichetta a causa di bad words ({len(texts_with_label_changes)} testi):")
    for item in texts_with_label_changes:
        print(f"Titolo: {item['Title']}")
        print(f"Testo originale: {item['Original Text']}")
        print(f"Parole inappropriate rilevate: {', '.join(item['Detected Bad Words'])}")
        print(f"Etichetta originale: {item['Original Label']}, Nuova etichetta: {item['New Label']}")
        print("-" * 80)


# # Flan T5 con badwords inglese

# In[ ]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Scarica risorse di NLTK per la tokenizzazione
nltk.download('punkt')

# Definizione del percorso dei dati
data_path = '/content/drive/My Drive/Esperimento_eng'

# Caricamento delle bad words in italiano
bad_words_path = '/content/drive/My Drive/badwords_eng_new.txt'

try:
    with open(bad_words_path, 'r', encoding='utf-8') as file:
        bad_words = set(line.strip() for line in file)
except Exception as e:
    print(f"Error loading bad words: {e}")
    bad_words = set()  # Definisci un set vuoto in caso di errore per evitare crash

# Funzione per rilevare bad words
def detect_bad_words(text, bad_words):
    words = set(word_tokenize(text.lower()))  # Tokenizza e abbassa il testo
    return words.intersection(bad_words)

# Funzione per caricare i dati
def load_data(data_path):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label_folder, label_index in label_dict.items():
        folder_path = os.path.join(data_path, label_folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(label_index)
                titles.append(filename)
    return texts, labels, titles

# Caricamento dei dati
texts, labels, titles = load_data(data_path)

# Tokenizzazione dei dati
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Creazione della classe Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Configurazione del Trainer
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

def compute_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions

    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}

# Dizionari per raccogliere i risultati medi
precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

accuracy_per_fold = []
macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

# Lista per memorizzare i testi in cui l'etichetta è cambiata
texts_with_label_changes = []

# Cross-Validation usando KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(texts)):
    print(f"Fold {fold + 1}")

    train_texts = [texts[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    train_labels = [labels[i] for i in train_index]
    test_labels = [labels[i] for i in test_index]
    train_titles = [titles[i] for i in train_index]
    test_titles = [titles[i] for i in test_index]

    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-small", num_labels=len(set(labels)))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = get_training_args(output_dir=f'./results_fold_{fold + 1}')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Avvio del training e della valutazione
    trainer.train()
    eval_results = trainer.evaluate()

    # Valutazione manuale sui dati di test
    predictions = trainer.predict(test_dataset)
    eval_results = compute_metrics(predictions)
    predicted_labels = np.argmax(predictions.predictions[0], axis=1)

    # Aggiungi il controllo delle bad words
    adjusted_labels = []
    detected_bad_words_list = []

    for i, text in enumerate(test_texts):
        detected_bad_words = detect_bad_words(text, bad_words)
        original_label = predicted_labels[i]
        if original_label == 0 and detected_bad_words:
            adjusted_labels.append(2)  # Cambia l'etichetta in "adulti" se ci sono bad words
            detected_bad_words_list.append((test_titles[i], detected_bad_words))

            # Memorizza i testi modificati
            texts_with_label_changes.append({
                "Title": test_titles[i],
                "Original Text": text,
                "Detected Bad Words": list(detected_bad_words),
                "Original Label": original_label,  # Etichetta originale (0 = bambini)
                "New Label": 2  # Nuova etichetta (2 = adulti)
            })
        else:
            adjusted_labels.append(original_label)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(test_labels, adjusted_labels, target_names=target_names, output_dict=True)

    # Somma i valori di precision, recall, f1 e support per ogni classe
    for class_name in target_names:
        precision_sum[class_name] += report[class_name]['precision']
        recall_sum[class_name] += report[class_name]['recall']
        f1_sum[class_name] += report[class_name]['f1-score']
        support_sum[class_name] += report[class_name]['support']

    # Somma per macro avg e weighted avg
    macro_avg_precision_sum += report['macro avg']['precision']
    macro_avg_recall_sum += report['macro avg']['recall']
    macro_avg_f1_sum += report['macro avg']['f1-score']

    weighted_avg_precision_sum += report['weighted avg']['precision']
    weighted_avg_recall_sum += report['weighted avg']['recall']
    weighted_avg_f1_sum += report['weighted avg']['f1-score']

    # Aggiungi l'accuratezza alla lista per calcolare la media finale
    accuracy_per_fold.append(report['accuracy'])

    # Stampa del classification report formattato
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        class_report = report[class_name]
        print(f"{class_name}\t{class_report['precision']:.2f}\t\t{class_report['recall']:.2f}\t{class_report['f1-score']:.2f}\t\t{class_report['support']}")

    print("\n\t\t\tSupport")
    print(f"Accuracy\t\t{report['accuracy']:.2f}\t{len(test_labels)}")
    print(f"Macro avg\t{report['macro avg']['precision']:.2f}\t\t{report['macro avg']['recall']:.2f}\t{report['macro avg']['f1-score']:.2f}\t\t{len(test_labels)}")
    print(f"Weighted avg\t{report['weighted avg']['precision']:.2f}\t\t{report['weighted avg']['recall']:.2f}\t{report['weighted avg']['f1-score']:.2f}\t\t{len(test_labels)}")

    # Stampa i testi in cui sono state rilevate bad words
    if detected_bad_words_list:
        print(f"Bad words rilevate nel fold {fold + 1}:")
        for title, bad_words in detected_bad_words_list:
            print(f"Titolo: {title}, Bad words: {bad_words}")

# Creazione della tabella riassuntiva finale
print("\n\n===== Tabella Riassuntiva dei Risultati Medi su 5 Fold =====")
print("\t\tPrecision\tRecall\tF1-score\tSupport")
for class_name in target_names:
    print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

# Stampa dell'accuratezza media finale
print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")

# Stampa dei testi in cui l'etichetta è stata cambiata a causa delle bad words
if texts_with_label_changes:
    print(f"\nTesti con cambiamento di etichetta a causa di bad words ({len(texts_with_label_changes)} testi):")
    for item in texts_with_label_changes:
        print(f"Titolo: {item['Title']}")
        print(f"Testo originale: {item['Original Text']}")
        print(f"Parole inappropriate rilevate: {', '.join(item['Detected Bad Words'])}")
        print(f"Etichetta originale: {item['Original Label']}, Nuova etichetta: {item['New Label']}")
        print("-" * 80)

