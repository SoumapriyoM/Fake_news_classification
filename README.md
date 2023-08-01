# Fake News Classification using Machine Learning
https://www.kaggle.com/competitions/fake-news/overview
This repository contains code and resources for Fake News Classification using Machine Learning. The project aims to build a robust model that can distinguish between real and fake news articles.

## Overview

Fake news has become a significant challenge in today's digital world, especially with the evolution of social media and online news platforms. The objective of this project is to develop a machine learning model that can accurately predict whether a given news article is genuine or fake. By addressing this challenge, we can promote information credibility and combat the spread of misinformation.

## Dataset

The dataset used for this project consists of news articles along with their corresponding labels (real or fake). It is available in the `data` folder and has been preprocessed to remove any missing values, making it ready for training.

## Data Preprocessing

Before training the model, the text data in the news articles undergoes preprocessing using powerful natural language processing (NLP) techniques. This involves tokenization, lemmatization, and stopwords removal to clean and prepare the text data for machine learning.

## Model Selection and Training

For the fake news classification task, we experiment with different machine learning algorithms, including Random Forest, Multinomial Naive Bayes, PassiveAggressive Classifier, and XGBoost. The models' performances are evaluated using metrics like accuracy and F1 score to select the best-performing one.

## Hyperparameter Tuning

To optimize the selected model's performance, we utilize RandomizedSearchCV to find the best hyperparameters. This fine-tunes the model and improves its accuracy and generalization.

## Results

The final model's performance is evaluated using metrics such as accuracy, F1 score, and confusion matrix. The results show the model's ability to distinguish between real and fake news articles.

## Kaggle Submission

The model developed in this project was submitted to a Kaggle competition on Fake News Classification. We achieved an impressive public score of 0.97948 and a private score of 0.97719. These scores demonstrate the model's strong generalization to both seen and unseen data, a crucial aspect in tackling the fake news problem.

## How to Use

To use this code for your own fake news classification task, you can clone the repository and run the Python scripts provided. The dataset and trained model will be available for experimentation. You can also fine-tune the model by adjusting hyperparameters and test it on your custom data.

## Conclusion

I am excited to share my journey in Fake News Classification and the achievements obtained in this Kaggle project. The model's ability to accurately classify real and fake news articles can play a vital role in promoting information authenticity and combating misinformation.

Feel free to explore the code and resources in this repository. If you have any suggestions or improvements, I welcome contributions and feedback. Together, let's advance technology to build a better and informed world! 🚀📚

---

_Stay tuned for more exciting projects and advancements in the world of data science and machine learning. Connect with me on LinkedIn or follow me here on GitHub to stay updated on my latest work._

Soumapriyo Mondal
Kaggle Score:
![Screenshot (1)](https://github.com/SoumapriyoM/Fake_news_classification/assets/103531896/9d62a1d7-f758-44d0-b0cd-09228717b9cc)

Word Cloud:
![download](https://github.com/SoumapriyoM/Fake_news_classification/assets/103531896/8c919f29-6bbc-4bf5-82eb-d55326c665c7)

