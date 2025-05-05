#!/usr/bin/env python
# coding: utf-8

# # RoBERTa testsplit

# In[ ]:


get_ipython().system('pip install accelerate -U')
get_ipython().system('pip install transformers[torch] -U')
get_ipython().system('pip install scikit-learn')


# In[ ]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/esperimento'


# In[ ]:


# Carica il tokenizer di RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# In[ ]:


# Classe Dataset
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


# In[ ]:


# Funzione per calcolare le metriche
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


# In[ ]:


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


# In[ ]:


# Caricamento e divisione dei dati
texts, labels, titles = load_data(data_path)
train_texts, test_texts, train_labels, test_labels, train_titles, test_titles = train_test_split(texts, labels, titles, test_size=0.2, random_state=42)

# Creazione dei dataset
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)


# In[ ]:


# Configurazione dei parametri di training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
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


# In[ ]:


# Carica il modello di RoBERTa per la classificazione di sequenze
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# In[ ]:


# Training e valutazione del modello
trainer.train()
trainer.save_model("./best_model")  # Salva il modello con la migliore accuratezza


# In[ ]:


# Valutazione manuale sui dati di test
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)


# In[ ]:


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


# # roBERTa per italiano (cased e uncased)

# In[ ]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/esperimento'

# Funzione per caricare i dati
def load_data(data_path, version):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label, index in label_dict.items():
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # Abbassare le lettere solo per la versione uncased
                if version == 'uncased':
                    text = text.lower()
                texts.append(text)
                labels.append(index)
                titles.append(filename)  # Salva il nome del file come titolo
    return texts, labels, titles

# Classe Dataset
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

# Funzione principale per eseguire il test cased o uncased
def test_model(version):
    if version == 'cased':
        # Tokenizer di RoBERTa-base (cased)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model_name = "roberta-base"
    else:
        # Tokenizer di RoBERTa-base (simulato uncased abbassando le lettere)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model_name = "roberta-base"

    # Caricamento dei dati
    texts, labels, titles = load_data(data_path, version)

    # Inizializza dizionari per raccogliere i risultati medi
    precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

    accuracy_per_fold = []
    macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
    weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

    fold = 1

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kfold.split(texts):
        print(f'\nFold {fold} - Versione {version}')
        fold += 1

        # Divisione in train e test per il fold corrente
        train_texts = [texts[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        # Creazione dei dataset
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # Configurazione dei parametri di training
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
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

        # Inizializzazione del modello e del trainer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Training del modello
        trainer.train()

        # Valutazione sul test del fold corrente
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Calcolo del classification report
        target_names = ['0_bambini', '1_ragazzi', '2_adulti']
        report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

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
    print(f"\nRisultati medi su 5 fold per la versione {version}:")
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

    # Media per Macro avg e Weighted avg
    print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
    print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

    # Stampa dell'accuratezza media finale
    print(f"\nAccuratezza media su 5 fold per la versione {version}: {np.mean(accuracy_per_fold):.2f}")

# Test per entrambe le versioni
test_model('cased')
test_model('uncased')


# # roBERTa italiano badwords

# In[ ]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
import string
import nltk

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Scarica risorse di NLTK per la tokenizzazione
nltk.download('punkt')

# Caricamento delle bad words
bad_words_path = '/content/drive/My Drive/bad_words.txt'

try:
    with open(bad_words_path, 'r', encoding='utf-8') as file:
        bad_words = set(line.strip() for line in file)
except Exception as e:
    print(f"Error loading bad words: {e}")

# Funzione per rilevare bad words
def detect_bad_words(text, bad_words):
    words = set(word_tokenize(text.lower()))  # Tokenizza e abbassa il testo
    return words.intersection(bad_words)

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/esperimento'

# Funzione per caricare i dati
def load_data(data_path, version):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label, index in label_dict.items():
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                if version == 'uncased':
                    text = text.lower()
                texts.append(text)
                labels.append(index)
                titles.append(filename)
    return texts, labels, titles

# Classe Dataset
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

# Funzione principale per eseguire il test cased o uncased con controllo delle bad words
def test_model(version, bad_words):
    if version == 'cased':
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model_name = "roberta-base"
    else:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model_name = "roberta-base"

    # Caricamento dei dati
    texts, labels, titles = load_data(data_path, version)

    # Inizializza dizionari per raccogliere i risultati medi
    precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

    accuracy_per_fold = []
    macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
    weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

    fold = 1

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kfold.split(texts):
        print(f'\nFold {fold} - Versione {version}')
        fold += 1

        # Divisione in train e test per il fold corrente
        train_texts = [texts[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        # Creazione dei dataset
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # Configurazione dei parametri di training
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
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

        # Inizializzazione del modello e del trainer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Training del modello
        trainer.train()

        # Valutazione sul test del fold corrente
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Aggiungi il controllo delle bad words
        adjusted_labels = []
        detected_bad_words_list = []

        for i, text in enumerate(test_texts):
            detected_bad_words = detect_bad_words(text, bad_words)
            original_label = predicted_labels[i]
            if original_label == 0 and detected_bad_words:
                adjusted_labels.append(2)  # Cambia l'etichetta in "adulti"
                detected_bad_words_list.append((titles[test_index[i]], detected_bad_words))
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

        # Stampa i testi in cui sono state rilevate bad words
        if detected_bad_words_list:
            print(f"Bad words rilevate nel fold {fold - 1}:")
            for title, bad_words in detected_bad_words_list:
                print(f"Titolo: {title}, Bad words: {bad_words}")

    # Media dei risultati per ogni classe
    print(f"\nRisultati medi su 5 fold per la versione {version}:")
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

    # Media per Macro avg e Weighted avg
    print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
    print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

    # Stampa dell'accuratezza media finale
    print(f"\nAccuratezza media su 5 fold per la versione {version}: {np.mean(accuracy_per_fold):.2f}")

# Test per entrambe le versioni con bad words check
test_model('cased', bad_words)
test_model('uncased', bad_words)


# # Roberta per inglese (cased e uncased)

# In[ ]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/Esperimento_eng'

# Funzione per caricare i dati
def load_data(data_path, version):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label, index in label_dict.items():
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # Abbassare le lettere solo per la versione uncased
                if version == 'uncased':
                    text = text.lower()
                texts.append(text)
                labels.append(index)
                titles.append(filename)  # Salva il nome del file come titolo
    return texts, labels, titles

# Classe Dataset
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

# Funzione principale per eseguire il test cased o uncased
def test_model(version):
    if version == 'cased':
        # Tokenizer di RoBERTa-base (cased)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model_name = "roberta-base"
    else:
        # Tokenizer di RoBERTa-base (simulato uncased abbassando le lettere)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model_name = "roberta-base"

    # Caricamento dei dati
    texts, labels, titles = load_data(data_path, version)

    # Inizializza dizionari per raccogliere i risultati medi
    precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

    accuracy_per_fold = []
    macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
    weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

    fold = 1

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kfold.split(texts):
        print(f'\nFold {fold} - Versione {version}')
        fold += 1

        # Divisione in train e test per il fold corrente
        train_texts = [texts[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        # Creazione dei dataset
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # Configurazione dei parametri di training
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
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

        # Inizializzazione del modello e del trainer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Training del modello
        trainer.train()

        # Valutazione sul test del fold corrente
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Calcolo del classification report
        target_names = ['0_bambini', '1_ragazzi', '2_adulti']
        report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

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
    print(f"\nRisultati medi su 5 fold per la versione {version}:")
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

    # Media per Macro avg e Weighted avg
    print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
    print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

    # Stampa dell'accuratezza media finale
    print(f"\nAccuratezza media su 5 fold per la versione {version}: {np.mean(accuracy_per_fold):.2f}")

# Test per entrambe le versioni
test_model('cased')
test_model('uncased')


# # roBERTa inglese badwords

# In[ ]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
import nltk

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Scarica risorse di NLTK per la tokenizzazione
nltk.download('punkt')

# Caricamento delle bad words
bad_words_path = '/content/drive/My Drive/badwords_eng_new.txt'

try:
    with open(bad_words_path, 'r', encoding='utf-8') as file:
        bad_words = set(line.strip() for line in file)
except Exception as e:
    print(f"Error loading bad words: {e}")

# Funzione per rilevare bad words
def detect_bad_words(text, bad_words):
    words = set(word_tokenize(text.lower()))  # Tokenizza e abbassa il testo
    return words.intersection(bad_words)

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/Esperimento_eng'

# Funzione per caricare i dati
def load_data(data_path):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label, index in label_dict.items():
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                labels.append(index)
                titles.append(filename)
    return texts, labels, titles

# Classe Dataset
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

# Funzione principale per eseguire il test cased con controllo delle bad words
def test_model(bad_words):
    # Tokenizer e modello RoBERTa-base
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model_name = "roberta-base"

    # Caricamento dei dati
    texts, labels, titles = load_data(data_path)

    # Inizializza dizionari per raccogliere i risultati medi
    precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

    accuracy_per_fold = []
    macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
    weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

    fold = 1

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kfold.split(texts):
        print(f'\nFold {fold} - RoBERTa cased')
        fold += 1

        # Divisione in train e test per il fold corrente
        train_texts = [texts[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        # Creazione dei dataset
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # Configurazione dei parametri di training
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
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

        # Inizializzazione del modello e del trainer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Training del modello
        trainer.train()

        # Valutazione sul test del fold corrente
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Aggiungi il controllo delle bad words
        adjusted_labels = []
        detected_bad_words_list = []

        for i, text in enumerate(test_texts):
            detected_bad_words = detect_bad_words(text, bad_words)
            original_label = predicted_labels[i]
            if original_label == 0 and detected_bad_words:
                adjusted_labels.append(2)  # Cambia l'etichetta in "adulti"
                detected_bad_words_list.append((titles[test_index[i]], detected_bad_words))
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

        # Stampa i testi in cui sono state rilevate bad words
        if detected_bad_words_list:
            print(f"Bad words rilevate nel fold {fold - 1}:")
            for title, bad_words in detected_bad_words_list:
                print(f"Titolo: {title}, Bad words: {bad_words}")

    # Media dei risultati per ogni classe
    print(f"\nRisultati medi su 5 fold per RoBERTa cased:")
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

    # Media per Macro avg e Weighted avg
    print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
    print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

    # Stampa dell'accuratezza media finale
    print(f"\nAccuratezza media su 5 fold per RoBERTa cased: {np.mean(accuracy_per_fold):.2f}")

# Test per la versione cased con bad words check
test_model(bad_words)


# # xml roberta per italiano (cased)

# In[ ]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/esperimento'

# Funzione per caricare i dati
def load_data(data_path, version):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label, index in label_dict.items():
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                labels.append(index)
                titles.append(filename)  # Salva il nome del file come titolo
    return texts, labels, titles

# Classe Dataset
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

# Funzione principale per eseguire il test cased o uncased
def test_model():
    # Tokenizer e modello di XLM-RoBERTa (multilingue)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model_name = "xlm-roberta-base"

    # Caricamento dei dati
    texts, labels, titles = load_data(data_path, version='multilingual')

    # Inizializza dizionari per raccogliere i risultati medi
    precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

    accuracy_per_fold = []
    macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
    weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

    fold = 1

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kfold.split(texts):
        print(f'\nFold {fold}')
        fold += 1

        # Divisione in train e test per il fold corrente
        train_texts = [texts[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        # Creazione dei dataset
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # Configurazione dei parametri di training
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
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

        # Inizializzazione del modello e del trainer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Training del modello
        trainer.train()

        # Valutazione sul test del fold corrente
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Calcolo del classification report
        target_names = ['0_bambini', '1_ragazzi', '2_adulti']
        report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

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
    print(f"\nRisultati medi su 5 fold:")
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

    # Media per Macro avg e Weighted avg
    print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
    print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

    # Stampa dell'accuratezza media finale
    print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")

# Test del modello
test_model()


# xlm roberta per inglese (cased)

# In[ ]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

# Montaggio di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Percorso della cartella dei dati su Google Drive
data_path = '/content/drive/My Drive/Esperimento_eng'

# Funzione per caricare i dati
def load_data(data_path, version):
    texts, labels, titles = [], [], []
    label_dict = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for label, index in label_dict.items():
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                labels.append(index)
                titles.append(filename)  # Salva il nome del file come titolo
    return texts, labels, titles

# Classe Dataset
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

# Funzione principale per eseguire il test cased o uncased
def test_model():
    # Tokenizer e modello di XLM-RoBERTa (multilingue)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model_name = "xlm-roberta-base"

    # Caricamento dei dati
    texts, labels, titles = load_data(data_path, version='multilingual')

    # Inizializza dizionari per raccogliere i risultati medi
    precision_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    recall_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    f1_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}
    support_sum = {class_name: 0 for class_name in ['0_bambini', '1_ragazzi', '2_adulti']}

    accuracy_per_fold = []
    macro_avg_precision_sum, macro_avg_recall_sum, macro_avg_f1_sum = 0, 0, 0
    weighted_avg_precision_sum, weighted_avg_recall_sum, weighted_avg_f1_sum = 0, 0, 0

    fold = 1

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kfold.split(texts):
        print(f'\nFold {fold}')
        fold += 1

        # Divisione in train e test per il fold corrente
        train_texts = [texts[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        # Creazione dei dataset
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # Configurazione dei parametri di training
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
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

        # Inizializzazione del modello e del trainer
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Training del modello
        trainer.train()

        # Valutazione sul test del fold corrente
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Calcolo del classification report
        target_names = ['0_bambini', '1_ragazzi', '2_adulti']
        report = classification_report(test_labels, predicted_labels, target_names=target_names, output_dict=True)

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
    print(f"\nRisultati medi su 5 fold:")
    print("\t\tPrecision\tRecall\tF1-score\tSupport")
    for class_name in target_names:
        print(f"{class_name}\t{(precision_sum[class_name]/5):.2f}\t\t{(recall_sum[class_name]/5):.2f}\t{(f1_sum[class_name]/5):.2f}\t\t{support_sum[class_name]}")

    # Media per Macro avg e Weighted avg
    print(f"\nMacro avg\t{(macro_avg_precision_sum/5):.2f}\t\t{(macro_avg_recall_sum/5):.2f}\t{(macro_avg_f1_sum/5):.2f}")
    print(f"Weighted avg\t{(weighted_avg_precision_sum/5):.2f}\t\t{(weighted_avg_recall_sum/5):.2f}\t{(weighted_avg_f1_sum/5):.2f}")

    # Stampa dell'accuratezza media finale
    print(f"\nAccuratezza media su 5 fold: {np.mean(accuracy_per_fold):.2f}")

# Test del modello
test_model()

