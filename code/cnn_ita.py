# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

!pip install numpy pandas scikit-learn tensorflow nltk tqdm

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Flatten
from tensorflow.keras.callbacks import LambdaCallback
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Verifica delle stopwords in italiano
italian_stopwords = set(stopwords.words('italian'))
print("Esempi di stopwords italiane:", list(italian_stopwords)[:10])

# Caricamento delle "bad words"
bad_words_path = '/content/drive/My Drive/bad_words.txt'

try:
    with open(bad_words_path, 'r', encoding='utf-8') as file:
        bad_words = set(line.strip() for line in file)
except Exception as e:
    print(f"Error loading bad words: {e}")

def detect_bad_words(text, bad_words):
    words = set(word_tokenize(text))
    found_bad_words = words.intersection(bad_words)
    return found_bad_words

def load_data(data_path):
    texts = []
    titles = []
    labels = []
    label_to_index = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for root, dirs, files in os.walk(data_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                titles.append(file_path)
                age_folder = os.path.basename(os.path.dirname(file_path))
                label = label_to_index.get(age_folder, '-')
                labels.append(label)
    return texts, titles, labels

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in italian_stopwords]
    return ' '.join(filtered_tokens)

# Funzione per la predizione con controllo delle "badwords"
def predict_with_bad_words_check_cnn(model, texts, titles, tokenizer, bad_words):
    sequences = tokenizer.texts_to_sequences([preprocess_text(text) for text in texts])
    X = pad_sequences(sequences, maxlen=100)
    predictions = model.predict(X)
    adjusted_predictions = []
    detected_bad_words_list = []

    for text, title, prediction in zip(texts, titles, predictions):
        detected_bad_words = detect_bad_words(preprocess_text(text), bad_words)
        predicted_label = np.argmax(prediction)
        if predicted_label == 0 and detected_bad_words:
            adjusted_predictions.append(2)  # Cambia l'etichetta in "adulti"
            detected_bad_words_list.append((title, text, detected_bad_words, predicted_label))
        else:
            adjusted_predictions.append(predicted_label)

    return adjusted_predictions, detected_bad_words_list

# Funzione per la predizione senza controllo delle "badwords"
def predict_without_bad_words_check_cnn(model, texts, titles, tokenizer):
    sequences = tokenizer.texts_to_sequences([preprocess_text(text) for text in texts])
    X = pad_sequences(sequences, maxlen=100)
    predictions = model.predict(X)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    return predicted_labels

# Caricamento dei dati
data_path = '/content/drive/My Drive/esperimento'
texts, titles, labels = load_data(data_path)

# Aggiungi stampe di debug
print(f"Numero di testi caricati: {len(texts)}")
print(f"Numero di etichette caricate: {len(labels)}")

# Controlla se i testi e le etichette sono stati caricati correttamente
if len(texts) == 0 or len(labels) == 0:
    print("Errore: Nessun testo o etichetta caricata. Verifica il percorso dei dati.")
else:
    # Preprocessing dei testi
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # Tokenizzazione e padding delle sequenze di testi
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(preprocessed_texts)
    sequences = tokenizer.texts_to_sequences(preprocessed_texts)
    X = pad_sequences(sequences, maxlen=100)
    y = np.array(labels)

    # Definizione dell'architettura della CNN
    def create_cnn_model():
        model = Sequential([
            Embedding(input_dim=10000, output_dim=64),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Flatten(),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true_with = []
    all_y_pred_with = []
    all_detected_bad_words = []

    all_y_true_without = []
    all_y_pred_without = []

    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        titles_train, titles_test = [titles[i] for i in train_index], [titles[i] for i in test_index]
        texts_train, texts_test = [texts[i] for i in train_index], [texts[i] for i in test_index]

        model = create_cnn_model()

        # Addestramento del modello CNN con tqdm per visualizzare la barra di avanzamento
        num_epochs = 10
        batch_size = 32
        with tqdm(total=num_epochs) as pbar:
            model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=[LambdaCallback(on_epoch_end=lambda _, __: pbar.update(1))])

        # Predizione sui dati di test con controllo delle "badwords"
        y_pred_with, detected_bad_words_list = predict_with_bad_words_check_cnn(model, texts_test, titles_test, tokenizer, bad_words)
        all_y_true_with.extend(y_test)
        all_y_pred_with.extend(y_pred_with)
        all_detected_bad_words.extend(detected_bad_words_list)

        # Predizione sui dati di test senza controllo delle "badwords"
        y_pred_without = predict_without_bad_words_check_cnn(model, texts_test, titles_test, tokenizer)
        all_y_true_without.extend(y_test)
        all_y_pred_without.extend(y_pred_without)

    # Report di classificazione dettagliato con controllo delle "badwords"
    report_with_badwords = classification_report(all_y_true_with, all_y_pred_with, target_names=['0_bambini', '1_ragazzi', '2_adulti'])
    print("Report di classificazione CNN con controllo delle badwords (k-fold cross-validation):\n", report_with_badwords)

    # Report di classificazione senza controllo delle "badwords"
    report_without_badwords = classification_report(all_y_true_without, all_y_pred_without, target_names=['0_bambini', '1_ragazzi', '2_adulti'])
    print("Report di classificazione CNN senza controllo delle badwords (k-fold cross-validation):\n", report_without_badwords)

    # Stampa degli errori di classificazione e cambiamenti di etichetta per le cartelle con file presenti
    def create_error_dataframe(titles, texts, true_labels, predictions):
        errors = np.where(np.array(true_labels) != np.array(predictions))[0]
        error_data = {'Title': [titles[idx] for idx in errors],
                      'True Label': [true_labels[idx] for idx in errors],
                      'Predicted Label': [predictions[idx] for idx in errors],
                      'Text': [texts[idx] for idx in errors]}
        error_df = pd.DataFrame(error_data)
        return error_df

    error_df_with_badwords = create_error_dataframe(titles, texts, all_y_true_with, all_y_pred_with)
    print(f"Errori di classificazione con controllo delle bad words:")
    print(error_df_with_badwords)

    # Stampa dei cambiamenti di etichetta a causa delle "badwords"
    print("Cambiamenti di etichetta a causa delle badwords:")
    for title, text, bad_words, original_prediction in all_detected_bad_words:
        print(f"Titolo: {title}")
        print(f"Testo: {text}")
        print(f"Badwords rilevate: {bad_words}")
        print(f"Predizione originale: {original_prediction}")
        print(f"Cambiata a: 2 (adulti)")
        print()

from google.colab import drive
drive.mount('/content/drive')

!pip install numpy pandas scikit-learn tensorflow nltk tqdm

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import LambdaCallback
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Verifica delle stopwords in italiano
italian_stopwords = set(stopwords.words('italian'))
print("Esempi di stopwords italiane:", list(italian_stopwords)[:10])

# Caricamento delle "bad words"
bad_words_path = '/content/drive/My Drive/bad_words.txt'

try:
    with open(bad_words_path, 'r', encoding='utf-8') as file:
        bad_words = set(line.strip() for line in file)
except Exception as e:
    print(f"Error loading bad words: {e}")

def detect_bad_words(text, bad_words):
    words = set(word_tokenize(text))
    found_bad_words = words.intersection(bad_words)
    return found_bad_words

def load_data(data_path):
    texts = []
    titles = []
    labels = []
    label_to_index = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    for root, dirs, files in os.walk(data_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                titles.append(file_path)
                age_folder = os.path.basename(os.path.dirname(file_path))
                label = label_to_index.get(age_folder, '-')
                labels.append(label)
    return texts, titles, labels

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in italian_stopwords]
    return ' '.join(filtered_tokens)

# Funzione per la predizione con controllo delle "badwords"
def predict_with_bad_words_check_cnn(model, texts, titles, tokenizer, bad_words):
    sequences = tokenizer.texts_to_sequences([preprocess_text(text) for text in texts])
    X = pad_sequences(sequences, maxlen=100)
    predictions = model.predict(X)
    adjusted_predictions = []
    detected_bad_words_list = []

    for text, title, prediction in zip(texts, titles, predictions):
        detected_bad_words = detect_bad_words(preprocess_text(text), bad_words)
        predicted_label = np.argmax(prediction)
        if predicted_label == 0 and detected_bad_words:
            adjusted_predictions.append(2)  # Cambia l'etichetta in "adulti"
            detected_bad_words_list.append((title, text, detected_bad_words, predicted_label))
        else:
            adjusted_predictions.append(predicted_label)

    return adjusted_predictions, detected_bad_words_list

# Funzione per la predizione senza controllo delle "badwords"
def predict_without_bad_words_check_cnn(model, texts, titles, tokenizer):
    sequences = tokenizer.texts_to_sequences([preprocess_text(text) for text in texts])
    X = pad_sequences(sequences, maxlen=100)
    predictions = model.predict(X)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    return predicted_labels

# Caricamento dei dati
data_path = '/content/drive/My Drive/esperimento'
texts, titles, labels = load_data(data_path)

# Aggiungi stampe di debug
print(f"Numero di testi caricati: {len(texts)}")
print(f"Numero di etichette caricate: {len(labels)}")

# Controlla se i testi e le etichette sono stati caricati correttamente
if len(texts) == 0 or len(labels) == 0:
    print("Errore: Nessun testo o etichetta caricata. Verifica il percorso dei dati.")
else:
    # Preprocessing dei testi
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # Tokenizzazione e padding delle sequenze di testi
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(preprocessed_texts)
    sequences = tokenizer.texts_to_sequences(preprocessed_texts)
    X = pad_sequences(sequences, maxlen=100)
    y = np.array(labels)

    # Definizione dell'architettura della CNN con Dropout
    def create_cnn_model():
        model = Sequential([
            Embedding(input_dim=10000, output_dim=64),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            GlobalMaxPooling1D(),
            Dropout(0.5),  # Dropout del 50% per prevenire overfitting
            Dense(64, activation='relu'),
            Dropout(0.5),  # Dropout aggiunto anche prima dello strato denso finale
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true_with = []
    all_y_pred_with = []
    all_detected_bad_words = []

    all_y_true_without = []
    all_y_pred_without = []

    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        titles_train, titles_test = [titles[i] for i in train_index], [titles[i] for i in test_index]
        texts_train, texts_test = [texts[i] for i in train_index], [texts[i] for i in test_index]

        model = create_cnn_model()

        # Addestramento del modello CNN con tqdm per visualizzare la barra di avanzamento
        num_epochs = 10
        batch_size = 32
        with tqdm(total=num_epochs) as pbar:
            model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=[LambdaCallback(on_epoch_end=lambda _, __: pbar.update(1))])

        # Predizione sui dati di test con controllo delle "badwords"
        y_pred_with, detected_bad_words_list = predict_with_bad_words_check_cnn(model, texts_test, titles_test, tokenizer, bad_words)
        all_y_true_with.extend(y_test)
        all_y_pred_with.extend(y_pred_with)
        all_detected_bad_words.extend(detected_bad_words_list)

        # Predizione sui dati di test senza controllo delle "badwords"
        y_pred_without = predict_without_bad_words_check_cnn(model, texts_test, titles_test, tokenizer)
        all_y_true_without.extend(y_test)
        all_y_pred_without.extend(y_pred_without)

    # Report di classificazione dettagliato con controllo delle "badwords"
    report_with_badwords = classification_report(all_y_true_with, all_y_pred_with, target_names=['0_bambini', '1_ragazzi', '2_adulti'])
    print("Report di classificazione CNN con controllo delle badwords (k-fold cross-validation):\n", report_with_badwords)

    # Report di classificazione senza controllo delle "badwords"
    report_without_badwords = classification_report(all_y_true_without, all_y_pred_without, target_names=['0_bambini', '1_ragazzi', '2_adulti'])
    print("Report di classificazione CNN senza controllo delle badwords (k-fold cross-validation):\n", report_without_badwords)

    # Stampa degli errori di classificazione e cambiamenti di etichetta per le cartelle con file presenti
    def create_error_dataframe(titles, texts, true_labels, predictions):
        errors = np.where(np.array(true_labels) != np.array(predictions))[0]
        error_data = {'Title': [titles[idx] for idx in errors],
                      'True Label': [true_labels[idx] for idx in errors],
                      'Predicted Label': [predictions[idx] for idx in errors],
                      'Text': [texts[idx] for idx in errors]}
        error_df = pd.DataFrame(error_data)
        return error_df

    error_df_with_badwords = create_error_dataframe(titles, texts, all_y_true_with, all_y_pred_with)
    print(f"Errori di classificazione con controllo delle bad words:")
    print(error_df_with_badwords)

    # Stampa dei cambiamenti di etichetta a causa delle "badwords"
    print("Cambiamenti di etichetta a causa delle badwords:")
    for title, text, bad_words, original_prediction in all_detected_bad_words:
        print(f"Titolo: {title}")
        print(f"Testo: {text}")
        print(f"Badwords rilevate: {bad_words}")
        print(f"Predizione originale: {original_prediction}")
        print(f"Cambiata a: 2 (adulti)")
        print()
