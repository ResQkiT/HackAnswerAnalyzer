import torch
import pandas as pd
from dao.ResponseObject import ResponseObject

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keybert import KeyBERT
import joblib

# Загрузка необходимых ресурсов NLTK

def week_analizer(file_path):
    return ResponseObject({});
  
def analyze_sentiment_for_dataframe(df, sentiment_pipeline):
    results = []
    for text in  df:  # Предполагаем, что текст находится в столбце 'text'
        text=text[:511]
        result = sentiment_pipeline(text)
        label, score = result[0]['label'], result[0]['score']
        results.append((text, label, score))
    return results

def deep_learn_analizer(file_path):
    
    df = pd.read_excel(file_path)
    df = df.iloc[:, 3]
    df = df.dropna()

    model_name = "blanchefort/rubert-base-cased-sentiment"  # Многоязычная версия DistilBERT
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


    sentiment_results = analyze_sentiment_for_dataframe(df, sentiment_pipeline)

    sentiment_df = pd.DataFrame(sentiment_results, columns=['text', 'label', 'score'])
    
    sentiment_df=sentiment_df["label"]
    d = {'neural': sentiment_df.value_counts()[0], 'positive': sentiment_df.value_counts()[1], 'negative': sentiment_df.value_counts()[2]}
    return ResponseObject(d)

def preprocess_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'\W', ' ', text)  # Удаление знаков препинания
    tokens = text.split()  # Токенизация
    tokens = [word for word in tokens if word not in stopwords.words('russian')]  # Удаление стоп-слов
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Лемматизация
    return ' '.join(tokens)

def data_prepocess(file_path):
    employee_comments = pd.read_excel(file_path)

    data = pd.DataFrame(employee_comments.apply(lambda row: ' '.join([str(row[i]) for i in range(2) if pd.notna(row[i])]), axis=1))
    # Удаление или замена пропущенных значений
    data.dropna(inplace=True)  # или data.fillna(value, inplace=True)

    # Удаление дублирующих записей
    data.drop_duplicates(inplace=True)

    #Подумать
    data[0] = data[0].apply(preprocess_text)  # Замените 'text_column' на ваше название колонки с текстом
    return data


# Функция для получения репрезентативного слова для каждого кластера
def get_representative_word(cluster_label, keybert_model, data):
    cluster_data = data[data['cluster_kmeans'] == cluster_label]
    if cluster_data.empty:
        return "Нет данных"
    
    # Извлекаем тексты из кластера
    texts = cluster_data[0].tolist()
    # Получаем ключевое слово с использованием KeyBERT
    keywords = keybert_model.extract_keywords(' '.join(texts), top_n=1)
    
    # Возвращаем первое ключевое слово
    return keywords[0][0] if keywords else "Нет ключевых слов"

def deep_learn_analizer2(file_path):
    kmeans = joblib.load('models/kmeans_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    # Инициализируем модель KeyBERT
    keybert_model = KeyBERT()

    data = data_prepocess(file_path)

    X = vectorizer.transform(data[0]).toarray()

    # Кластеризация с использованием KMeans
    n_cluster = 8 # Задайте количество кластеров
    
    data['cluster_kmeans'] = kmeans.predict(X)
    

    # Получение репрезентативного слова для каждого кластера
    cluster_summary = {}

    for i in range(n_cluster):
        representative_word = get_representative_word(i, keybert_model, data)
        count = len(data[data['cluster_kmeans'] == i])  # Количество элементов в кластере
        cluster_summary[f"Cluster {i} ({representative_word})"] = count
    
    return ResponseObject(cluster_summary)

    