import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

st.title("Sentiment Analysis Dashboard")

df = pd.read_csv("/content/review data.csv")

st.write("Sample Data")
st.dataframe(df.head())

x = df['text']
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

model = LogisticRegression()

model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

label_counts = pd.Series(y_pred).value_counts()

fig, ax = plt.subplots(figsize=(8, 6))

label_counts.plot(kind='barh', ax=ax)

ax.set_xlabel('Number of Predictions')
ax.set_ylabel('Sentiment')
ax.set_title('Sentiment Distribution')

st.pyplot(fig)
