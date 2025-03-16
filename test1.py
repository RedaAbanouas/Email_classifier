import stanza
import pandas as pd
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from collections import Counter
import re

nlp = stanza.Pipeline("fr", processors="tokenize,mwt,pos,lemma")

df = pd.read_excel("Projet_SPAM.xlsx")
X = df[df["type"] == "Spam"]

stop_words = set(stopwords.words('french'))
text = " ".join(X["email"].astype(str))
doc = nlp(text)

lemmatized_text = " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])
wrds = re.findall(r'\b\w+\b', lemmatized_text.lower())

filtered_words = [i for i in wrds if i not in stop_words]

word_counts = Counter(filtered_words)
print(word_counts.most_common(15))

def count_uppercase(text):
    return len(re.findall(r'[A-Z]', text))
df['nbre_maj'] = 0;
df['nbre_maj'] = df['email'].apply(count_uppercase)

