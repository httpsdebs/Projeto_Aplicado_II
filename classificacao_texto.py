import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import warnings
warnings.filterwarnings("ignore")

print("Carregando dados...")
categories = ['rec.sport.baseball', 'sci.space', 'comp.graphics', 'talk.politics.mideast']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
df['target_name'] = df['target'].apply(lambda x: newsgroups.target_names[x])

print(f"Total de amostras: {len(df)}")
print("Categorias:", newsgroups.target_names)

print("\nAnálise exploratória dos dados:")
print(df['target_name'].value_counts())

df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(10,6))
sns.boxplot(x='target_name', y='text_length', data=df)
plt.title("Distribuição do tamanho dos textos por categoria")
plt.ylabel("Número de caracteres")
plt.xlabel("Categoria")
plt.show()

X = df['text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTamanho do conjunto de treino: {len(X_train)}")
print(f"Tamanho do conjunto de teste: {len(X_test)}")

print("\nVetorizando os textos com TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words='english',  
    max_df=0.7,           
    min_df=5,             
    ngram_range=(1,2)     
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Número de features após vetorização: {X_train_tfidf.shape[1]}")

models = {
    "Naive Bayes": MultinomialNB(),
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "SVM Linear": LinearSVC()
}

results = {}

for name, model in models.items():
    print(f"\nTreinando modelo: {name}")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia no teste: {acc:.4f}")
    results[name] = {
        'model': model,
        'accuracy': acc,
        'y_pred': y_pred
    }

best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\nMelhor modelo: {best_model_name} com acurácia {results[best_model_name]['accuracy']:.4f}")

y_pred_best = results[best_model_name]['y_pred']

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_best, target_names=newsgroups.target_names))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=newsgroups.target_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title(f"Matriz de Confusão - {best_model_name}")
plt.show()

novos_textos = [
    "The spacecraft successfully landed on Mars and sent back images.",
    "The baseball game last night was exciting and the home team won.",
    "The new graphics card improves rendering performance significantly.",
    "The political situation in the Middle East is very complex."
]

novos_tfidf = vectorizer.transform(novos_textos)
predicoes = best_model.predict(novos_tfidf)

print("\nPredições para novos textos:")
for texto, pred in zip(novos_textos, predicoes):
    print(f"Texto: \"{texto}\"")
    print(f"Categoria prevista: {newsgroups.target_names[pred]}")
    print("-" * 50)

import joblib

joblib.dump(best_model, 'modelo_texto_classificacao.pkl')
joblib.dump(vectorizer, 'vetorizador_tfidf.pkl')

print("\nModelo e vetor TF-IDF salvos em disco para uso futuro.")

print("\nProjeto de classificação de texto concluído com sucesso!")