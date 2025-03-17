import stanza
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import emoji

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

def normalize(text):
    doc = nlp(text)
    lemmatized_text = " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])
    return lemmatized_text.lower()

def count_uppercase(text):
    return len(re.findall(r'[A-Z]', text))

list_var = ['fichier', 'ici', 'cliquer', 'télécharger', 'offre', 'maintenant', 'recevoir', 'lien', 'ouvrir', 'compte']
df['nbre_maj'] = 0
df[list_var] = 0;
df['nb_emojis'] = 0;
df['euro_dollar'] = 0

def count_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)
def euro_dollar(text):
    return len(re.findall(r'[\€\$]', text))

df['nbre_maj'] = df['email'].apply(count_uppercase)
df["nb_emojis"] = df["email"].apply(count_emojis)
df["euro_dollar"] = df["email"].apply(euro_dollar)
df["email_normalized"] = df["email"].apply(normalize)
for var in list_var:
    df[var] = df["email_normalized"].apply(lambda text: text.count(var))

ordre = ['email', 'email_normalized', 'nbre_maj', 'nb_emojis',
         'euro_dollar','fichier', 'ici', 'cliquer', 'télécharger',
         'offre', 'maintenant', 'recevoir', 'lien', 'ouvrir', 'compte', 'type']
df = df[ordre]

df.to_excel("Projet_SPAM_modifie.xlsx", index=False)