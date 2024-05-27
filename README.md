# Sentiment Analysis on Twitter Data

[![Binder](https://mybinder.org/badge_logo.svg)](https://notebooks.gesis.org/binder/jupyter/user/andrewzgheib-ml-timent-analysis-nahwr9di/lab/tree/main.ipynb)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewzgheib/ML-Sentiment-Analysis/blob/main/main.ipynb)
[![Kaggle](https://camo.githubusercontent.com/1398db766d8ca60e6f296aac9ac429e344705c6c2c7e1ceb024230aac69fd6be/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667)](https://www.kaggle.com/code/andrewzgheib/notebookadd7decf9c/edit)

This project focuses on building a sentiment analysis model using machine learning techniques to classify tweets as positive or negative. The model is trained on a large dataset of tweets with corresponding sentiment labels.

## Dataset

The dataset used in this project is the "Sentiment140" dataset from Kaggle, which contains 1.6 million tweets. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Dependencies

The following Python libraries are required to run the code:

- pandas
- re
- pickle
- nltk
- sklearn

## Approach

The main steps involved in this project are as follows:

1. **Data Preprocessing**: The dataset is loaded into a pandas DataFrame, and the text data is cleaned and preprocessed. This includes removing non-alphabetic characters, tokenizing the tweets, and stemming the words using NLTK's PorterStemmer.

2. **Feature Extraction**: The preprocessed text data is converted into numerical features using TfidfVectorizer from scikit-learn. This creates a sparse matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features.

3. **Model Training**: The dataset is split into training and testing sets (80% for training and 20% for testing). A Logistic Regression model is trained on the training data using the TF-IDF features as input and the sentiment labels as targets.

4. **Model Evaluation**: The trained model is evaluated on both the training and testing data by calculating the accuracy score.

5. **Model Saving**: The trained model is saved as a pickle file for future use.

6. **Model Testing**: The saved model is loaded and tested on new examples from the testing set to ensure it is working as expected.

## Usage

To run the code, simply open the `main.ipynb` notebook and execute the cells sequentially. The notebook is self-contained and includes all the necessary code and explanations.

## Results

The trained Logistic Regression model achieves an accuracy of approximately 79.87% on the training data and 77.67% on the testing data. These results demonstrate the effectiveness of the model in classifying tweets as positive or negative based on their textual content.
