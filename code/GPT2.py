#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install accelerate -U')
get_ipython().system('pip install transformers[torch] -U')
get_ipython().system('pip install scikit-learn')


# In[ ]:


import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load data from your Google Drive
# Replace this path with your own if you're running locally or from a different Drive
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

# Creazione del tokenizer GPT-2 e aggiunta del pad_token
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Verifica se il pad_token è già definito
if tokenizer.pad_token is None:
    # Aggiungi un token di padding al tokenizer
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

# Funzione di tokenizzazione
def tokenize_function(text_list):
    return tokenizer(text_list, padding="longest", truncation=True, max_length=512, return_tensors="pt")

# Tokenizzazione dei dati
encodings = tokenize_function(texts)

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

# Creazione del dataset
dataset = TextDataset(encodings, labels)

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funzione per calcolare le metriche
def compute_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        logits = eval_pred.predictions[0]
    else:
        logits = eval_pred.predictions

    labels = eval_pred.label_ids

    print("Shape of logits:", logits.shape)
    print("Shape of labels:", labels.shape)

    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}

# Configurazione del Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
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

# Creazione della cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}")

    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=test_subset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    predictions = trainer.predict(test_subset)
    fold_metrics = compute_metrics(predictions)
    results.append(fold_metrics)
    print(f"Fold {fold + 1} Results: {fold_metrics}")

# Media delle metriche sui fold
mean_accuracy = np.mean([result['accuracy'] for result in results])
print(f"Mean Accuracy over {kf.get_n_splits()} folds: {mean_accuracy}")

# Calcolo del classification report finale (solo per l'ultimo fold)
predicted_labels = np.argmax(predictions.predictions, axis=1)
report = classification_report([dataset.labels[i] for i in test_index], predicted_labels, target_names=['0_bambini', '1_ragazzi', '2_adulti'], output_dict=True)

# Stampa del classification report formattato
print("\t\tPrecision\tRecall\tF1-score\tSupport")
for class_name in ['0_bambini', '1_ragazzi', '2_adulti']:
    class_report = report[class_name]
    print(f"{class_name}\t{class_report['precision']:.2f}\t\t{class_report['recall']:.2f}\t{class_report['f1-score']:.2f}\t\t{class_report['support']}")

print("\n\t\t\tSupport")
print(f"Accuracy\t\t{report['accuracy']:.2f}\t{len(test_index)}")
print(f"Macro avg\t{report['macro avg']['precision']:.2f}\t\t{report['macro avg']['recall']:.2f}\t{report['macro avg']['f1-score']:.2f}\t\t{len(test_index)}")
print(f"Weighted avg\t{report['weighted avg']['precision']:.2f}\t\t{report['weighted avg']['recall']:.2f}\t{report['weighted avg']['f1-score']:.2f}\t\t{len(test_index)}")

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


# In[3]:


import os
import torch
from torch.utils.data import Dataset, Subset
from transformers import GPT2ForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Montaggio di Google Drive (se usi Colab)
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

# Creazione del tokenizer GPT-2 e aggiunta del pad_token
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Verifica se il pad_token è già definito
if tokenizer.pad_token is None:
    # Aggiungi un token di padding al tokenizer
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

# Funzione di tokenizzazione
def tokenize_function(text_list):
    return tokenizer(text_list, padding="longest", truncation=True, max_length=512, return_tensors="pt")

# Tokenizzazione dei dati
encodings = tokenize_function(texts)

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

# Creazione del dataset
dataset = TextDataset(encodings, labels)

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funzione per calcolare le metriche
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Configurazione del Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
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

# Creazione della cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}")

    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=test_subset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    predictions = trainer.predict(test_subset)

    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = [labels[i] for i in test_index]

    # Aggiungi il controllo delle bad words e modifica delle etichette
    adjusted_labels = []
    for i, text_idx in enumerate(test_index):
        text = texts[text_idx]
        detected_bad_words = detect_bad_words(text, bad_words)
        original_label = predicted_labels[i]

        if original_label == 0 and detected_bad_words:  # Cambia l'etichetta se ci sono bad words
            adjusted_labels.append(2)  # Cambia l'etichetta in "adulti"
            texts_with_label_changes.append({
                "Title": titles[text_idx],
                "Original Text": text,
                "Detected Bad Words": list(detected_bad_words),
                "Original Label": original_label,
                "New Label": 2
            })
        else:
            adjusted_labels.append(original_label)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(true_labels, adjusted_labels, target_names=target_names, output_dict=True)

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

# Media delle metriche sui fold
print("\n===== Tabella Riassuntiva dei Risultati Medi su 5 Fold =====")
print("\t\tPrecision\tRecall\tF1-score\tSupport")
for class_name in target_names:
    print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

# Stampa dell'accuratezza media finale
print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")

# Stampa dei testi in cui l'etichetta è stata cambiata
if texts_with_label_changes:
    print(f"\nTesti con cambiamento di etichetta a causa delle bad words ({len(texts_with_label_changes)} testi):")
    for item in texts_with_label_changes:
        print(f"Titolo: {item['Title']}")
        print(f"Testo originale: {item['Original Text']}")
        print(f"Parole inappropriate rilevate: {', '.join(item['Detected Bad Words'])}")
        print(f"Etichetta originale: {item['Original Label']}, Nuova etichetta: {item['New Label']}")
        print("-" * 80)


# In[4]:


import os
import torch
from torch.utils.data import Dataset, Subset
from transformers import GPT2ForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Montaggio di Google Drive (se usi Colab)
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

# Creazione del tokenizer GPT-2 e aggiunta del pad_token
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Verifica se il pad_token è già definito
if tokenizer.pad_token is None:
    # Aggiungi un token di padding al tokenizer
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

# Funzione di tokenizzazione
def tokenize_function(text_list):
    return tokenizer(text_list, padding="longest", truncation=True, max_length=512, return_tensors="pt")

# Tokenizzazione dei dati
encodings = tokenize_function(texts)

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

# Creazione del dataset
dataset = TextDataset(encodings, labels)

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funzione per calcolare le metriche
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Configurazione del Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
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

# Creazione della cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}")

    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=test_subset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    predictions = trainer.predict(test_subset)

    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = [labels[i] for i in test_index]

    # Aggiungi il controllo delle bad words e modifica delle etichette
    adjusted_labels = []
    for i, text_idx in enumerate(test_index):
        text = texts[text_idx]
        detected_bad_words = detect_bad_words(text, bad_words)
        original_label = predicted_labels[i]

        if original_label == 0 and detected_bad_words:  # Cambia l'etichetta se ci sono bad words
            adjusted_labels.append(2)  # Cambia l'etichetta in "adulti"
            texts_with_label_changes.append({
                "Title": titles[text_idx],
                "Original Text": text,
                "Detected Bad Words": list(detected_bad_words),
                "Original Label": original_label,
                "New Label": 2
            })
        else:
            adjusted_labels.append(original_label)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(true_labels, adjusted_labels, target_names=target_names, output_dict=True)

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

# Media delle metriche sui fold
print("\n===== Tabella Riassuntiva dei Risultati Medi su 5 Fold =====")
print("\t\tPrecision\tRecall\tF1-score\tSupport")
for class_name in target_names:
    print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

# Stampa dell'accuratezza media finale
print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")

# Stampa dei testi in cui l'etichetta è stata cambiata
if texts_with_label_changes:
    print(f"\nTesti con cambiamento di etichetta a causa delle bad words ({len(texts_with_label_changes)} testi):")
    for item in texts_with_label_changes:
        print(f"Titolo: {item['Title']}")
        print(f"Testo originale: {item['Original Text']}")
        print(f"Parole inappropriate rilevate: {', '.join(item['Detected Bad Words'])}")
        print(f"Etichetta originale: {item['Original Label']}, Nuova etichetta: {item['New Label']}")
        print("-" * 80)

