#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')
# Installazione delle dipendenze necessarie
get_ipython().system('pip install accelerate -U')
get_ipython().system('pip install transformers[torch]')


# In[ ]:


# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/esperimento'
import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from huggingface_hub import login

# Imposta il token di autenticazione
os.environ["HF_TOKEN"] = "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"

# Effettua il login
login(token="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")

# Inizializza il tokenizzatore
tokenizer = AutoTokenizer.from_pretrained("sapienzanlp/Minerva-350M-base-v1.0")

# Definisci il token di padding
tokenizer.pad_token = tokenizer.eos_token

# Classe Dataset per gestire i dati
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Funzione per calcolare le metriche
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Funzione per caricare i dati
def load_data(data_path):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label, index in label_dict.items():
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(index)
                titles.append(filename)  # Salva il nome del file come titolo
    return texts, labels, titles

# Caricamento dei dati
texts, labels, titles = load_data(data_path)

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

    # Creazione dei dataset
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)

    # Impostazioni di training
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold + 1}',
        num_train_epochs=10,
        per_device_train_batch_size=2,  # Adatta la dimensione del batch alla tua GPU
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # Carica il modello Minerva per la classificazione delle sequenze
    model = AutoModelForSequenceClassification.from_pretrained("sapienzanlp/Minerva-350M-base-v1.0", num_labels=3)

    # Imposta il token di padding nel modello
    model.config.pad_token_id = tokenizer.pad_token_id

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Addestramento e valutazione del modello
    trainer.train()
    trainer.save_model(f"./best_model_fold_{fold + 1}")

    # Valutazione manuale sui dati di test
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    # Calcolo del classification report
    target_names = ['0_bambini', '1_ragazzi', '2_adulti']
    report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

    # Stampa del classification report
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

