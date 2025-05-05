
# Safeguarding Minors: a Study on Automatic Age-based Classification of Online Multimedia Content using Text Transcripts 

This repository contains code and documentation for the PhD research project by Antonietta Paone on automatically classifying online multimedia content by age appropriateness using NLP and machine learning techniques.

## 📂 Repository Structure

/code/              → Python and notebook scripts (Colab-based)
/dictionaries/      → Custom dictionaries for violence, drugs, and bad language
/annotations/       → Annotation guidelines
/corpora/           → Annotated English and Italian corpora, organized by age group folders (e.g., 0_bambini, 1_ragazzi, 2_adulti)
/semantic_analysis/ → Scripts for semantic similarity computation and network generation (Word2Vec, cosine similarity, Gephi-ready output)
/README.md          → This file

## 📑 Description

The goal is to support age-based classification of online content (TED, YouTube, etc.) to protect minors. Transcripts were analyzed using classical machine learning models, deep neural networks, and transformer-based language models. A bad words filter was used to flag and correct misclassified adult content.

## 🧠 Models Used

### Classical ML (TF-IDF + GridSearchCV)
- SVM
- Naive Bayes
- Decision Tree
- Random Forest
- K-Nearest Neighbors

### Deep Learning
- CNN (with dropout and max-pooling)
- LSTM / BiLSTM

### Transformer Models (via Hugging Face)
- BERT / mBERT / DistilBERT
- RoBERTa / ELECTRA
- AlBERTo, GilBERTo, UmBERTo

### LLMs (Large Language Models)
- GPT-2
- Flan-T5
- Minerva

All models were evaluated using 5-fold cross-validation. Predictions were further corrected using a custom “bad words” dictionary, adjusting child-labeled texts containing inappropriate language.

## 🧪 Datasets

The datasets are structured in three folders:
0_bambini/     → Children-appropriate transcripts  
1_ragazzi/     → Teens-appropriate content  
2_adulti/      → Adult or unsuitable content  

The corpora were compiled from a mix of existing and manually collected resources:

- **TED2020** (Reimers & Gurevych, 2020): 196 English and 202 Italian TED Talk transcripts, originally translated by volunteers. Selected for thematic variety and quality.
- **TVOKids** (EN): Transcripts of preschool and school-age videos manually collected from tvokids.com.
- **YouTube Kids** (IT): Subtitles extracted from videos aimed at different age groups (Preschool to 12 y/o).
- **OpenSubtitles** (EN/IT): Horror and adult-genre subtitles, filtered by language.
- **YouTube Poop (YTP)** (EN/IT): Informal adult content scraped from YouTube, manually reviewed and adjusted.

All texts were annotated for age suitability and thematic indicators (e.g., bad language, drugs, violence) using a custom tagset based on AGCOM's classification guidelines.

The final dataset includes:
- **English**: 446 texts
- **Italian**: 514 texts
  
⚠️ A portion of the texts was extracted using web scraping techniques, strictly for academic research purposes only. Users are kindly requested to use the data for non-commercial, research-related activities only and to respect content copyright.

If you use or reference the TED2020 portion of the corpus, please cite:  
Reimers, N., & Gurevych, I. (2020). Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

## 📚 Dictionaries

Three domain-specific dictionaries were created manually and semi-automatically:
- **Violence Dictionary**
- **Drug Dictionary**
- **Bad Language Dictionary**
- *Bad Words list*

⚠️ The **English dictionaries** were developed in the context of the work by Maisto et al.  
Please cite: Maisto, Alessandro, Giandomenico Martorelli, Antonietta Paone, and Serena Pelosi (2021a), “Automatic Classification and Rating of Videogames Based on Dialogues Transcript Files”, in International Conference on Emerging Internetworking, Data & Web Technologies, Springer, pp. 301-312.

⚠️ The NRC Emotion Intensity Lexicon (Mohammad & Turney, 2013; Mohammad et al., 2018) was also extended and translated into Italian to assist with emotional analysis. The translated and modified version is not included in this repository due to redistribution restrictions. Researchers interested in this resource may contact the author for academic use only, or refer to the original version available at:
[[https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm](https://saifmohammad.com/WebPages/AffectIntensity.htm)]

### 🔧 Word2Vec Model Training

The Word2Vec models used in this project were trained from scratch using the Gensim library:

- **Italian Word2Vec**: trained on the PAISÀ corpus (Lyding et al., 2014)
- **English Word2Vec**: trained on the British National Corpus (Leech 1992)

Key parameters used:
- vector size: 300
- window: 5
- min_count: 2
- sg: 1 (skip-gram model)

⚠️ The corpora and trained models are not included in this repository due to copyright restrictions.  
Researchers can recreate the models using public versions of PAISÀ and BNC.

## ▶️ Execution (Colab-based)

All notebooks were run using Google Colab. Key dependencies:
pip install numpy pandas scikit-learn tensorflow nltk transformers tqdm

Mount your Google Drive inside notebooks:
from google.colab import drive  
drive.mount('/content/drive')

Models were trained and evaluated with automatic reporting and post-prediction filtering.

## ⚖️ License

 MIT License (see LICENSE file)
---

For academic or research use, please cite this repository or the original thesis.
