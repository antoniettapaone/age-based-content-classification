
# Safeguarding Minors: Age-Based Classification of Online Multimedia Content

This repository contains code and documentation for the PhD research project by Antonietta Paone on automatically classifying online multimedia content by age appropriateness using NLP and machine learning techniques.

## 📂 Repository Structure

/code/              → Python and notebook scripts (Colab-based)
/dictionaries/      → Custom dictionaries for violence, drugs, and bad language
/annotations/       → Annotation guidelines
/corpora/           → Annotated English and Italian corpora, organized by age group folders (e.g., 0_bambini, 1_ragazzi, 2_adulti)
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

⚠️ Original corpora are not included due to copyright restrictions.  
Only format descriptions or metadata may be provided.

## 📚 Dictionaries

Three domain-specific dictionaries were created manually and semi-automatically:
- **Violence Dictionary**
- **Drug Dictionary**
- **Bad Language Dictionary**

The NRC Emotion Intensity Lexicon was also extended and translated into Italian to assist with emotional analysis.

## ▶️ Execution (Colab-based)

All notebooks were run using Google Colab. Key dependencies:
pip install numpy pandas scikit-learn tensorflow nltk transformers tqdm

Mount your Google Drive inside notebooks:
from google.colab import drive  
drive.mount('/content/drive')

Models were trained and evaluated with automatic reporting and post-prediction filtering.

## ⚖️ License

- Code: MIT License (see LICENSE file)
- Dictionaries & annotations: Creative Commons BY-NC 4.0 (non-commercial use only)
- Corpora: Not included – request access if needed

---

For academic or research use, please cite this repository or the original thesis.
