import spacy

nlp = spacy.load("fr_core_news_sm")  # Charger le modèle français
text = "Les enfants jouaient dans le parc et mangeaient des glaces."

doc = nlp(text)
lemmatized_text = " ".join([token.lemma_ for token in doc])

print(lemmatized_text)

