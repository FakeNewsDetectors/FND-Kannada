
# Fake News Detection in Kannada
This repository contains the code for the project "Intersection of Machine Learning, Deep Learning and Transformers to Combat Fake News in Kannada Language". The code and the proposed fake news detection pipeline can be effectively utilized to tackle the problem of fake news in various linguistic contexts.

This repository proposes fake news detection system in Kannada by implementing many Machine Learning (SVM, Decision Trees etc.), Deep Learning (CNN, Bi-LSTM), Transformers (KannadaBERT, GPT-2 etc.) and Ensemble (XGBoost etc.) algorithms in combination with various word embeddings (fastText, TF-IDF etc.).


## Paper Abstract
Fake news has developed into a pervasive issue causing enormous damage to both individuals and society. It is of utmost importance that fake news is identified and combated in all languages including those with fewer resources, like Kannada. We created a dataset of 2800 labelled instances and developed a fake news detection system using machine learning, deep learning and transformer architectures like KannadaBERT and GPT-2 in
combination with various word embedding techniques like fastText and TF-IDF, which achieved an accuracy of 92%. This framework can be widely applied to address fake news in other languages.

![Proposed architecture](https://github.com/FakeNewsDetectors/FND-Kannada/blob/main/Images/pipeline.png)
*Figure 1: Overview of the proposed architecture*


## Implementation
### Dataset
A dataset comprising 2800 instances of news headlines was created by scraping Kannada news websites including Prajavani, OneIndia, Factly, PublicTV etc. using Beautiful Soup library and Octoparse.

[combined_dataset.csv](https://github.com/FakeNewsDetectors/FND-Kannada/blob/main/data/v2/combined_dataset.csv): Dataset used in this study for fake news detection in Kannada.



### Code
- [Scaraping](https://github.com/FakeNewsDetectors/FND-Kannada/tree/main/codes/Scraping): Directory containing codes for scraping fake news headlines from Asianet, Factly, OneIndia and Prajavani websites.
- [Cleaning and Preprocessing](https://github.com/FakeNewsDetectors/FND-Kannada/tree/main/codes/Cleaning%20and%20pre-processing): Directory containing codes for data cleaning and train-test split of dataset.
- [Models](https://github.com/FakeNewsDetectors/FND-Kannada/tree/main/codes/Models): Directory containing codes for Machine Learning and Deep Learning models for fake news detection. Hybrid models directory contains codes for models which use different techniques for feature extraction (word embeddings) and classification. Vanilla models directory contains codes where the same model extracts embeddings and perform classification. 


### Machine Learning Algorithms
- Logistic Regression
- Support Vector Machine
- Passive Aggressive Classifier
- Decision Trees

### Ensemble Algorithms
- Voting Classifier of Logistic Regression and Support Vector Machine
- XGBoost
- Random Forest

### Deep Learning Algorithms
- Convolutional Neural Network (CNN)
- Bi-directional Long Short Term Memory (Bi-LSTM)
- KannadaBERT
- GPT-2
- mBERT
- Distil-mBERT

### Word Embeddings 
- KannadaBERT
- GPT-2
- mBERT
- Distil-mBERT
- IndicBERT
- fastText
- IndicFT
- TF-IDF
- One-hot encoding

### Requirements, Libraries and Tools
- Python
- Octoparse
- Beautiful Soup
- Pandas
- NumPy
- Scikit-learn 
- Tensor-Flow 
- Keras

## Results
Models were evaluated using various metrics, including accuracy, confusion matrix and True Positive Rate (TPR) on the test set.  

*Table 1: Classification results*
| Classification Model | Word Embeddings | Accuracy | Precision | Recall | f1-score | TPR  |
|----------------------|-----------------|----------|-----------|--------|----------|------|
| KannadaBERT          | KannadaBERT     | 0.92     | 0.92      | 0.91   | 0.91     | 0.93 |
| GPT-2                | GPT-2           | 0.91     | 0.94      | 0.82   | 0.88     | 0.94 |
| CNN                  | KannadaBERT     | 0.90     | 0.90      | 0.90   | 0.90     | 0.92 |
| CNN                  | mBERT           | 0.88     | 0.89      | 0.88   | 0.88     | 0.87 |
| CNN                  | IndicBERT       | 0.87     | 0.87      | 0.87   | 0.87     | 0.84 |
| mBERT                | mBERT           | 0.87     | 0.86      | 0.86   | 0.86     | 0.82 |
| Distil-mBERT         | Distil-mBERT    | 0.85     | 0.84      | 0.85   | 0.85     | 0.82 |
| CNN                  | IndicBERT       | 0.84     | 0.85      | 0.84   | 0.84     | 0.89 |
| CNN                  | fastText        | 0.82     | 0.82      | 0.82   | 0.81     | 0.86 |
| BiLSTM               | fastText        | 0.82     | 0.82      | 0.82   | 0.82     | 0.75 |
| BiLSTM               | One-hot encoding| 0.81     | 0.81      | 0.81   | 0.81     | 0.75 |
| CNN                  | One-hot encoding| 0.81     | 0.81      | 0.81   | 0.81     | 0.77 |
| BiLSTM               | IndicFT         | 0.76     | 0.80      | 0.76   | 0.76     | 0.64 |
| Ensemble model (LR, SVM)| TF-IDF        | 0.76     | 0.78      | 0.76   | 0.74     | 0.85 |
| Logistic Regression  | TF-IDF          | 0.75     | 0.79      | 0.75   | 0.73     | 0.90 |
| SVM                  | TF-IDF          | 0.75     | 0.75      | 0.75   | 0.73     | 0.78 |
| BiLSTM               | IndicBERT       | 0.72     | 0.75      | 0.72   | 0.72     | 0.63 |
| Random Forest        | TF-IDF          | 0.72     | 0.71      | 0.82   | 0.71     | 0.68 |
| Passive Aggressive Classifier | TF-IDF  | 0.71     | 0.71      | 0.71   | 0.71     | 0.68 |
| XGBoost              | TF-IDF          | 0.71     | 0.73      | 0.71   | 0.68     | 0.78 |
| Decision Trees       | TF-IDF          | 0.66     | 0.67      | 0.66   | 0.66     | 0.58 |
 

## Contact-Info
Please feel free to contact us for any questions. We will be happy to help.
- Email: artiarya@pes.edu
- GitHub: https://github.com/FakeNewsDetectors







