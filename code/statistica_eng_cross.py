# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

!pip install scikit-learn
!pip install nltk
!pip install pandas

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

def load_data_from_folders(base_path):
    labels = {'0_bambini': 0, '1_ragazzi': 1, '2_adulti': 2}
    texts = []
    y = []
    titles = []
    folder_labels = []

    try:
        for label, idx in labels.items():
            folder_path = os.path.join(base_path, label)
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                        y.append(idx)
                        titles.append(filename)
                        folder_labels.append(label)
    except Exception as e:
        print(f"Error loading data: {e}")
    return texts, y, titles, folder_labels

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Caricamento delle "bad words"
bad_words_path = '/content/drive/My Drive/badwords_eng_new.txt'

try:
    with open(bad_words_path, 'r', encoding='utf-8') as file:
        bad_words = set(line.strip() for line in file)
except Exception as e:
    print(f"Error loading bad words: {e}")

def create_error_dataframe(titles, texts, true_labels, predictions):
    errors = np.where(true_labels != predictions)[0]
    error_data = {'Title': [titles[idx] for idx in errors],
                  'True Label': [true_labels[idx] for idx in errors],
                  'Predicted Label': [predictions[idx] for idx in errors],
                  'Text': [texts[idx] for idx in errors]}
    error_df = pd.DataFrame(error_data)
    return error_df

def detect_bad_words(text, bad_words):
    words = set(word_tokenize(text))
    found_bad_words = words.intersection(bad_words)
    return found_bad_words

def predict_with_bad_words_check(model, texts, titles, vectorizer, bad_words):
    tfidf_vectors = vectorizer.transform(texts)
    predictions = model.predict(tfidf_vectors)
    adjusted_predictions = []
    detected_bad_words_list = []

    for text, title, prediction in zip(texts, titles, predictions):
        detected_bad_words = detect_bad_words(preprocess_text(text), bad_words)
        if prediction == 0 and detected_bad_words:
            adjusted_predictions.append(2)  # Cambia l'etichetta in "adulti"
            detected_bad_words_list.append((title, text, detected_bad_words))
        else:
            adjusted_predictions.append(prediction)

    return adjusted_predictions, detected_bad_words_list

def cross_val_predict_with_bad_words(model, X, y, texts, titles, vectorizer, bad_words, cv=5):
    y_pred = cross_val_predict(model, X, y, cv=cv)
    adjusted_y_pred = []
    detected_bad_words_list = []

    for text, title, prediction in zip(texts, titles, y_pred):
        detected_bad_words = detect_bad_words(preprocess_text(text), bad_words)
        if prediction == 0 and detected_bad_words:
            adjusted_y_pred.append(2)  # Cambia l'etichetta in "adulti"
            detected_bad_words_list.append((title, text, detected_bad_words))
        else:
            adjusted_y_pred.append(prediction)

    return np.array(adjusted_y_pred), detected_bad_words_list

def cross_validate_model(model, X, y, texts, titles, vectorizer, bad_words, apply_bad_words_check, cv=5):
    if apply_bad_words_check:
        y_pred, detected_bad_words_list = cross_val_predict_with_bad_words(model, X, y, texts, titles, vectorizer, bad_words, cv=cv)
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv)
        detected_bad_words_list = []

    report = classification_report(y, y_pred, target_names=['0_bambini', '1_ragazzi', '2_adulti'], output_dict=True)
    return report, detected_bad_words_list, y_pred

def evaluate_models_with_cv(models_params, X, y, texts, titles, vectorizer, bad_words):
    results_with_badwords = {}
    results_without_badwords = {}

    for name, (model, param_grid) in models_params.items():
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_

        print(f"{name} Best Params: {grid_search.best_params_}")

        # Predizione con cross-validation e controllo delle bad words
        report_with_badwords, detected_bad_words_list_with, y_pred_with = cross_validate_model(best_model, X, y, texts, titles, vectorizer, bad_words, apply_bad_words_check=True, cv=5)
        results_with_badwords[name] = {
            'best_params': grid_search.best_params_,
            'report': report_with_badwords,
            'detected_bad_words_list': detected_bad_words_list_with,
            'y_pred': y_pred_with
        }

        # Predizione con cross-validation senza controllo delle bad words
        report_without_badwords, _, y_pred_without = cross_validate_model(best_model, X, y, texts, titles, vectorizer, bad_words, apply_bad_words_check=False, cv=5)
        results_without_badwords[name] = {
            'best_params': grid_search.best_params_,
            'report': report_without_badwords,
            'y_pred': y_pred_without
        }

    return results_with_badwords, results_without_badwords

# Funzione per stampare il report
def print_classification_report(report, title):
    print(f"{title}")
    print("=" * len(title))
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{label}")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-score: {metrics['f1-score']:.2f}")
            print(f"  Support: {metrics['support']}")
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(f"Macro avg")
    print(f"  Precision: {report['macro avg']['precision']:.2f}")
    print(f"  Recall: {report['macro avg']['recall']:.2f}")
    print(f"  F1-score: {report['macro avg']['f1-score']:.2f}")
    print(f"Weighted avg")
    print(f"  Precision: {report['weighted avg']['precision']:.2f}")
    print(f"  Recall: {report['weighted avg']['recall']:.2f}")
    print(f"  F1-score: {report['weighted avg']['f1-score']:.2f}")

# Caricamento dei dati e preprocessing
base_path = '/content/drive/My Drive/Esperimento_eng'
texts, labels, titles, folder_labels = load_data_from_folders(base_path)
texts = [preprocess_text(text) for text in texts]
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(texts)

# Definizione dei modelli e parametri per GridSearchCV
models_params = {
    'SVM': (SVC(random_state=42), {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100]}),
    'Naive Bayes': (MultinomialNB(), {'alpha': [0.5, 1.0, 1.5]}),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 4, 10, 20, 30]}),
    'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200, 300], 'max_features': ['auto', 'sqrt', 'log2']}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']})
}

# Valutazione dei modelli con cross-validation sui dati completi
results_with_badwords, results_without_badwords = evaluate_models_with_cv(models_params, X_tfidf, labels, texts, titles, vectorizer, bad_words)

# Stampa dei risultati per ogni modello con cross-validation
for name, result in results_with_badwords.items():
    print_classification_report(result['report'], f"Risultati per {name} con controllo delle bad words (cross-validation)")
    # Stampa delle bad words individuate e dei cambiamenti di etichetta
    print(f"{name} Bad words individuate e cambiamenti di etichetta:")
    for title, text, bad_words in result['detected_bad_words_list']:
        print(f"Title: {title}\nDetected Bad Words: {bad_words}\nChanged to '2_adulti'\n")
    # Stampa degli errori di classificazione
    error_df_with = create_error_dataframe(titles, texts, labels, result['y_pred'])
    print(f"Errori di classificazione con controllo delle bad words per {name} (cross-validation):")
    print(error_df_with)

for name, result in results_without_badwords.items():
    print_classification_report(result['report'], f"Risultati per {name} senza controllo delle bad words (cross-validation)")
    # Stampa degli errori di classificazione
    error_df_without = create_error_dataframe(titles, texts, labels, result['y_pred'])
    print(f"Errori di classificazione senza controllo delle bad words per {name} (cross-validation):")
    print(error_df_without)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

# Usa il tuo vettorizzatore esistente e il dataset preprocessato
X_tfidf = vectorizer.fit_transform(texts)

# Dividi i dati in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

# Riduci le dimensioni del dataset a due componenti principali per la visualizzazione
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())

# Addestra il modello K-Nearest Neighbors sui dati ridotti
knn = KNeighborsClassifier(n_neighbors=9, weights='uniform')
knn.fit(X_train_pca, y_train)

# Crea una meshgrid per rappresentare le regioni decisionali
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predici le classi per ogni punto della meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Crea una figura
plt.figure(figsize=(10, 6))

# Colori per le classi
colors = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFAAAA'])  # Verde per l'etichetta 0, blu per l'etichetta 1, rosso per l'etichetta 2

# Visualizza le regioni decisionali
plt.contourf(xx, yy, Z, alpha=0.3, cmap=colors)

# Visualizza i punti di addestramento e di test
scatter_train = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=colors, edgecolor='k', s=50, marker='o', label='Training data')
scatter_test = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=colors, edgecolor='k', s=50, marker='^', label='Test data')

# Aggiungi una legenda dettagliata
train_labels = plt.scatter([], [], marker='o', color='#000000', edgecolor='k', label='Training data')
test_labels = plt.scatter([], [], marker='^', color='#000000', edgecolor='k', label='Test data')
class_0 = plt.scatter([], [], marker='o', color='#AAFFAA', edgecolor='k', label='Class 0')
class_1 = plt.scatter([], [], marker='o', color='#AAAAFF', edgecolor='k', label='Class 1')
class_2 = plt.scatter([], [], marker='o', color='#FFAAAA', edgecolor='k', label='Class 2')

plt.legend(handles=[train_labels, test_labels, class_0, class_1, class_2], loc='upper left')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Regions of K-Nearest Neighbors Classifier')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

# Genera un dataset bidimensionale sintetico
X_synth, y_synth = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
X_synth += 2 * np.random.RandomState(42).uniform(size=X_synth.shape)

# Addestra il modello di Random Forest sul dataset sintetico
clf_synth = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf_synth.fit(X_synth, y_synth)

# Crea una meshgrid per rappresentare le regioni decisionali
x_min, x_max = X_synth[:, 0].min() - 1, X_synth[:, 0].max() + 1
y_min, y_max = X_synth[:, 1].min() - 1, X_synth[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predici le classi per ogni punto della meshgrid
Z = clf_synth.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Crea una figura
plt.figure(figsize=(10, 6))

# Colori per le classi
colors = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# Visualizza le regioni decisionali
plt.contourf(xx, yy, Z, alpha=0.4, cmap=colors)

# Visualizza i punti di addestramento
scatter = plt.scatter(X_synth[:, 0], X_synth[:, 1], c=y_synth, cmap=colors, edgecolor='k', s=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions of Random Forest Classifier')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()

# Dopo GridSearchCV per il Decision Tree
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), {'max_depth': [None, 4, 10, 20, 30]}, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_dt.fit(X_tfidf, labels)

# Salva il miglior modello
best_decision_tree_model = grid_search_dt.best_estimator_

# Aggiungi una nuova cella nel Colab per visualizzare l'albero decisionale
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualizza l'albero decisionale
plt.figure(figsize=(20,10))
plot_tree(best_decision_tree_model, feature_names=vectorizer.get_feature_names_out(), class_names=['0_bambini', '1_ragazzi', '2_adulti'], filled=True, rounded=True)
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Supponiamo di avere le etichette vere e predette per un modello
true_labels = [0, 1, 2, 2, 0, 1, 2, 1, 0, 2]  # Esempio di etichette vere
predicted_labels_with_badwords = [0, 1, 2, 0, 0, 1, 2, 1, 0, 2]  # Esempio di etichette predette con bad words
predicted_labels_without_badwords = [0, 1, 2, 0, 0, 1, 2, 2, 0, 2]  # Esempio di etichette predette senza bad words

# Creazione della matrice di confusione
cm_with_badwords = confusion_matrix(true_labels, predicted_labels_with_badwords)
cm_without_badwords = confusion_matrix(true_labels, predicted_labels_without_badwords)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_with_badwords, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Matrice di Confusione con Bad Words')
ax[0].set_xlabel('Predicted Labels')
ax[0].set_ylabel('True Labels')

sns.heatmap(cm_without_badwords, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title('Matrice di Confusione senza Bad Words')
ax[1].set_xlabel('Predicted Labels')
ax[1].set_ylabel('True Labels')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

def count_texts_with_bad_words(texts, folder_labels, bad_words):
    bad_words_texts_counts = {label: 0 for label in set(folder_labels)}
    total_texts_counts = {label: 0 for label in set(folder_labels)}

    for text, label in zip(texts, folder_labels):
        detected_bad_words = detect_bad_words(preprocess_text(text), bad_words)
        if detected_bad_words:
            bad_words_texts_counts[label] += 1
        total_texts_counts[label] += 1

    return bad_words_texts_counts, total_texts_counts

def plot_bad_words_distribution(bad_words_texts_counts, total_texts_counts):
    labels = list(bad_words_texts_counts.keys())
    counts_with_badwords = list(bad_words_texts_counts.values())
    total_counts = list(total_texts_counts.values())
    counts_without_badwords = [total_counts[i] - counts_with_badwords[i] for i in range(len(total_counts))]

    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(labels))

    bars1 = ax.bar(index, counts_with_badwords, bar_width, label='Texts with Bad Words')
    bars2 = ax.bar(index + bar_width, counts_without_badwords, bar_width, label='Texts without Bad Words')

    ax.set_xlabel('Folders')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of Texts with and without Bad Words per Folder')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# Percorso alla cartella dei dati
base_path = '/content/drive/My Drive/Esperimento_eng'
texts, labels, titles, folder_labels = load_data_from_folders(base_path)

# Conteggio dei testi con e senza bad words per ogni cartella
bad_words_texts_counts, total_texts_counts = count_texts_with_bad_words(texts, folder_labels, bad_words)

# Visualizzazione della distribuzione dei testi con e senza bad words
plot_bad_words_distribution(bad_words_texts_counts, total_texts_counts)
