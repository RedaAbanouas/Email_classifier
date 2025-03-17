import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import emoji
import pickle

df = pd.read_excel('Post_nlp_database.xlsx')
X = df.iloc[:, 2:15]
y = df.iloc[:, 15]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred, labels=['Spam', 'Non spam'])
print("Confusion Matrix:\n", cm)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)