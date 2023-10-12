import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import string
from collections import Counter
import gensim

# Cuisines list
CUISINES = [
    "Mexican", "Italian", "Indian", "Thai", "Korean", "French", "Latin American", "Chinese", "Japanese", "Spanish"
]

# Grab stop words
stop_words = list(stopwords.words("english"))

# Initialize term vector
term_vec = []

# Loop through all cuisines
for cuisine in CUISINES:
    # Read instructions CSV
    df_instr = pd.read_csv(f"data/instructions/individual/{cuisine}_instructions.csv")
    instructions = ""

    for i, row in df_instr.iterrows():
        # Extract instruction
        instruction = row["Instructions"]

        # Replace the redundant word 'directions'
        instruction = instruction.replace("directions", "")

        # Append to instructions
        instructions += instruction

    # Remove punctuation, then tokenize documents
    punc = re.compile( '[%s]' % re.escape(string.punctuation))
    instructions = instructions.lower()
    instructions = punc.sub("", instructions)

    # Add to term vector
    term_vec.append(nltk.word_tokenize(instructions))

# Remove stop words
for i in range(len(term_vec)):
    term_list = []

    # Add to new list as long as it's not a stop word
    for term in term_vec[i]:
        if term not in stop_words:
            term_list.append(term)

    # Replace with stop-wordless string
    term_vec[i] = term_list

# Convert term vectors into gensim dictionary
dict = gensim.corpora.Dictionary(term_vec)

corp = []
for x in term_vec:
    corp.append(dict.doc2bow(x))

# Create TFIDF vectors based on term vectors bag-of-word corpora
tfidf_model = gensim.models.TfidfModel(corp)

tfidf = []
for x in corp:
    tfidf.append(tfidf_model[x])

tfidf_np = np.array(tfidf)

tfidf_svd = np.linalg.svd(tfidf_np)

print(tfidf_svd)