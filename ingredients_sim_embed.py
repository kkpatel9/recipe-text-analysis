import numpy as np
import pandas as pd
import spacy

# Cuisines list
CUISINES = [
    "Mexican", "Italian", "Indian", "Thai", "Korean", "French", "Latin American", "Chinese", "Japanese", "Spanish"
]

nlp = spacy.load("en_core_web_md")

# Create initial NLP models of full document text
doc_nlp = []

# Loop through all cuisines
for cuisine in CUISINES:
    # Read instructions CSV
    df_ing = pd.read_csv(f"data/ingredients/{cuisine}_ingredients.csv")
    ingredients = df_ing.iloc[0,1]

    doc_nlp.append(nlp(ingredients))

# Strip punctuation, numbers, stop words
doc_strip = []
for i, d_nlp in enumerate(doc_nlp):
    doc_strip.append([tok.text for tok in d_nlp if (tok.is_alpha & (not tok.is_stop))])
    doc_strip[-1] = ' '.join(doc_strip[-1])

# Re-compute NLP on stripped documents
doc_strip_nlp = []
for d in doc_strip:
    doc_strip_nlp.append(nlp(d))

# Build similarity matrix
sim_mat = np.diag([ 1.0 ] * len(doc_strip_nlp))
for i in range(len(doc_strip_nlp)-1):
    for j in range(i+1, len(doc_strip_nlp)):
        sim_mat[i][j] = doc_strip_nlp[i].similarity(doc_strip_nlp[j])
        sim_mat[j][i] = sim_mat[i][j]

# Print Similarity matrix
for i in range(len(sim_mat)):
    print(CUISINES[i] + " Cuisine:")

    for j in range(len(sim_mat[i])):
        print("Similarity with " + CUISINES[j] + " cuisine " + ("%.3f " % sim_mat[i][j]))

    print()