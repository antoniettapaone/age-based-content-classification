"""
Semantic Similarity - Full Pipeline (English)
Author: Antonietta Paone

This script reproduces the full semantic similarity pipeline applied to the English corpus:
1. Preprocessing and lemmatization with SpaCy
2. Extraction and semantic expansion of frequent terms using Word2Vec (trained on BNC)
3. Cosine similarity computation across documents
4. Export of similarity matrix and edge list for Gephi visualization

Note:
- For academic use only.
- Ensure you adjust paths and filenames to match your local or Colab environment.
- This script assumes the availability of preprocessed texts and a trained Word2Vec model.
"""

# -*- coding: utf-8 -*-
"""Parole_Vicine_eng.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SXX3jpGusx6_zgI2_v9vlrY_2dGALphm
"""


# Scarica risorse di NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from google.colab import drive

"""**Creazione corpus unico**"""

import os
import shutil

# Path delle cartelle principali
root_dir = '/content/drive/My Drive/Esperimento_eng'
destination_dir = '/content/drive/My Drive/Analisi_Semantica_2/corpus_eng'

# Creare la cartella di destinazione se non esiste
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterare attraverso le sottocartelle di root_dir
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        # Iterare attraverso i file nella sottocartella
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(file_path):
                # Copiare il file nella cartella di destinazione
                shutil.copy(file_path, destination_dir)

print(f"Tutti i file sono stati copiati in {destination_dir}")

"""**1. Preprocessiamo il testo**"""

import os
import spacy
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import re

# Inizializza il modello di Spacy per l'inglese
nlp = spacy.load('en_core_web_sm')


# Funzione di preprocessamento
def preprocess(text):
    # Converte il testo in minuscolo
    text = text.lower()
    # Sostituisce gli apostrofi con uno spazio
    text = text.replace("'", " ")
    # Rimuove punteggiatura, caratteri speciali e numeri
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    # Tokenizza il testo
    tokens = word_tokenize(text)
    # Rimuove le stopwords inglesi
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatizzazione
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return lemmas

# Percorsi delle cartelle di input e output
cartella_input = '/content/drive/My Drive/Anlisi_Semantica_2/corpus_eng'
cartella_output = '/content/drive/My Drive/Analisi_Semantica_2/corpus_eng_preprocessato'

# Assicurati che la cartella di output esista
os.makedirs(cartella_output, exist_ok=True)

# Preprocessamento dei file
files = os.listdir(cartella_input)
for file in tqdm(files):
    path_input = os.path.join(cartella_input, file)
    path_output = os.path.join(cartella_output, file)

    with open(path_input, 'r', encoding='utf-8') as f:
        text = f.read()
        preprocessed_text = preprocess(text)

    with open(path_output, 'w', encoding='utf-8') as f:
        f.write(" ".join(preprocessed_text))

print("Preprocessing completed.")

"""**2. Otteniamo le 10 parole più simili alle 100 parole più frequenti di ogni script**"""

import os
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm  # Importa la libreria tqdm per la barra di avanzamento

# Carica il modello Word2Vec addestrato sui dati PaSa
model_path = '/content/drive/My Drive/Analisi_Semantica_2/word2vec_model/word2vec_model_eng'  # Modifica questo percorso se necessario
model = Word2Vec.load(model_path)

# Percorsi delle cartelle di input e output
cartella_input = '/content/drive/My Drive/Analisi_Semantica_2/corpus_eng_preprocessato'  # Modifica questo percorso se necessario

cartella_output = '/content/drive/My Drive/Analisi_Semantica_2/Parole_vicine_inglese'
# Assicurati che la cartella di output esista
os.makedirs(cartella_output, exist_ok=True)

# Lista dei file nella cartella di input preprocessato
files = os.listdir(cartella_input)

for file in tqdm(files):  # Utilizza tqdm per creare la barra di avanzamento
    # Path del file di input preprocessato
    path_input = os.path.join(cartella_input, file)

    # Path del file di output
    nome_file = os.path.splitext(file)[0]  # Nome del file senza estensione
    nome_file_output = nome_file + "_output.txt"
    path_output = os.path.join(cartella_output, nome_file_output)

    # Carica il testo tokenizzato e lemmatizzato
    with open(path_input, "r", encoding="utf-8") as f:
        testo_tokenizzato = [line.split() for line in f]

    # Calcola le frequenze delle parole nel testo
    frequenze = Counter([parola for frase in testo_tokenizzato for parola in frase])

    # Trova le 100 parole più frequenti
    parole_frequenti = [parola for parola, frequenza in frequenze.most_common(100)]

    # Trova le parole semanticamente più vicine per ogni parola frequente
    parole_vicine = {}
    for parola in parole_frequenti:
        try:
            parole_simili = model.wv.most_similar(parola, topn=10)
            parole_vicine[parola] = [parola_simile for parola_simile, _ in parole_simili]
        except KeyError:
            parole_vicine[parola] = []

    # Salva le parole semanticamente vicine in un file di output
    with open(path_output, "w", encoding="utf-8") as output_file:
        for parola, parole_simili in parole_vicine.items():
            output_file.write(f"Parola: {parola}\n")
            output_file.write(f"Parole semanticamente vicine: {', '.join(parole_simili)}\n")
            output_file.write("\n")

print("Processing and saving of semantic words completed.")

"""**3. Trasformiamo i file ottenuti in liste di parole**"""

# Percorso alla cartella con i file originali in Google Drive
folder_path = '/content/drive/My Drive/Analisi_Semantica_2/Parole_vicine_inglese'
# Percorso alla cartella di output (verrà creata se non esiste)
output_folder = '/content/drive/My Drive/Analisi_Semantica_2/Lista_inglese'
os.makedirs(output_folder, exist_ok=True)

# Funzione per estrarre parole principali e vicine da un file
def parse_and_flatten(file_path):
    words_list = []
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            if line.startswith("Parola:"):
                main_word = line.split(":")[1].strip()
                words_list.append(main_word)
            elif line.startswith("Parole semanticamente vicine:"):
                similar_words = line.split(":")[1].strip().split(", ")
                words_list.extend(similar_words)
    return words_list

# Processa tutti i file `.txt` nella cartella
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        word_list = parse_and_flatten(file_path)
        output_file_path = os.path.join(output_folder, f"{file_name}_list.txt")

        # Scrivi la lista di parole su un nuovo file
        with open(output_file_path, 'w', encoding='utf8') as output_file:
            output_file.write("\n".join(word_list))

print("Tutti i file sono stati processati e salvati come liste.")

"""**4. Calcolo del Cosine Similarity**

Per ogni script, vogliamo creare un vettore che indica se ciascuna parola del vocabolario comune è presente (1) o assente (0) nel testo dello script. Questi vettori vengono poi memorizzati in un DataFrame per poter calcolare la similarità tra gli script. Per farlo si seguiranno queste fasi:

1. Impostazione delle directory.
2. Creazione del vocabolario comune: tutti i file nella crtella delle liste vengono letti e per ogni file di lista, vengono lette le parole, trasformate in minuscolo e aggiunte al set di parole uniche, il set viene ordinato in ordine alfabetico e viene creato il vocabolario.
3. Creazione del DataFrame dei vettori di presenza: viene inizializzato un DataFrame con colonne corrispondenti alle parole del vocabolario.
Per ogni file di lista nella directory viene costruito il nome del file dello script corrispondente, viene caricato il contenuto dello script e creato un vettore di presenza che rappresenta la presenza (1) o l'assenza (0) di ciascuna parola del vocabolario nel testo. Il vettored i presenza viene aggiunto poi al DataFrame.
4. Calcolo della similarità coseno: utilizza il *cosine_similarity* per calcolare la similarità tra i vettori di presenza di tutti gli script.
5. Creazione della matrice di similarità in un file Excel nella directory di output.
"""

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Percorsi delle directory contenenti gli script e le liste
scripts_folder = '/content/drive/My Drive/Analisi_Semantica_2/corpus_eng'
lists_folder = '/content/drive/My Drive/Analisi_Semantica_2/Lista_inglese'
output_folder = '/content/drive/My Drive/Analisi_Semantica_2/Risultati_inglese'
os.makedirs(output_folder, exist_ok=True)

# Crea un set per accumulare tutte le parole uniche
all_words = set()

# Processa tutti i file di lista per costruire il vocabolario comune
for list_file_name in os.listdir(lists_folder):
    if list_file_name.endswith("_list.txt"):
        list_path = os.path.join(lists_folder, list_file_name)
        with open(list_path, 'r', encoding='utf8') as word_file:
            words = [line.strip().lower() for line in word_file]
            all_words.update(words)

# Ordina il vocabolario e convertilo in una lista
vocabulary = sorted(all_words)

# Stampa il vocabolario per confermare (opzionale)
print("Vocabolario comune:", vocabulary)

# Funzione per caricare il testo dallo script originale
def load_script(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        return file.read().lower()

# Funzione per creare un vettore di presenza con un vocabolario fisso
def create_presence_vector(script_text, vocabulary):
    script_words = set(script_text.split())
    return [1 if word in script_words else 0 for word in vocabulary]

# Inizializza un DataFrame per memorizzare i vettori di presenza
results_df = pd.DataFrame(columns=vocabulary)

# Processa ogni file di lista e crea un vettore di presenza
for list_file_name in os.listdir(lists_folder):
    if list_file_name.endswith("_list.txt"):
        base_name = list_file_name.replace("_output.txt_list.txt", "")
        script_file_name = base_name + ".txt"
        script_path = os.path.join(scripts_folder, script_file_name)

        # Carica il testo dello script
        try:
            script_content = load_script(script_path)
        except FileNotFoundError:
            print(f"File di script {script_file_name} non trovato, salto...")
            continue

        # Crea il vettore di presenza per questo script
        presence_vector = create_presence_vector(script_content, vocabulary)

        # Aggiungi il vettore al DataFrame con l'indice corrispondente
        results_df.loc[base_name] = presence_vector

        # Stampa un messaggio di stato
        print(f"Processato: {base_name}")

# Calcola la similarità coseno tra tutti i vettori
similarity_matrix = cosine_similarity(results_df)

# Salva la matrice di similarità in un file Excel
output_path = os.path.join(output_folder, 'similarity_results.xlsx')
similarity_df = pd.DataFrame(similarity_matrix, index=results_df.index, columns=results_df.index)
similarity_df.to_excel(output_path, engine='openpyxl')

print("Risultati di similarità salvati in:", output_path)

"""**5. Trasformiamo la matrice in un file csv per la rappresentazione con Gephi**"""

import os

output_folder = '/content/drive/My Drive/Analisi_Semantica_2/Risultati_inglese'

# Esporta la matrice di similarità come file CSV per Gephi
output_path_csv = os.path.join(output_folder, 'similarity_results.csv')

# Converte la matrice di similarità in un formato 'Edge List'
edges = []
for i, source in enumerate(results_df.index):
    for j, target in enumerate(results_df.index):
        if i != j:  # Ignora self-loops
            weight = similarity_matrix[i, j]
            if weight > 0:  # Considera solo archi con peso positivo
                edges.append([source, target, weight])

# Crea un DataFrame per gli archi e salvalo come CSV
edges_df = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
edges_df.to_csv(output_path_csv, index=False)

print(f"Matrice di similarità per Gephi salvata in: {output_path_csv}")
