from bs4 import BeautifulSoup
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
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
    df_ing = pd.read_csv(f"data/ingredients/{cuisine}_ingredients.csv")
    ingredients = df_ing.iloc[0,1]

    # Remove punctuation, then tokenize documents
    punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
    ingredients = ingredients.lower()
    ingredients = punc.sub("", ingredients)

    # Add to term vector
    term_vec.append(nltk.word_tokenize(ingredients))

# Remove stop words
for i in range(len(term_vec)):
    term_list = []

    # Add to new list as long as it doesn't have a digit nor is a stop word
    for term in term_vec[i]:
        if not any(char.isdigit() for char in term) and term not in stop_words:
            term_list.append(term)

    # Replace with stop-wordless string
    term_vec[i] = term_list

# Porter stem remaining terms
porter = nltk.stem.porter.PorterStemmer()

for i in range(len(term_vec)):
    for j in range(len(term_vec[i])):
        term_vec[i][j] = porter.stem(term_vec[i][j])

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

# Create pairwise cuisine similarity index
n = len(dict)
index = gensim.similarities.SparseMatrixSimilarity(tfidf_model[corp], num_features = n)

# #  Print TFIDF vectors
# for i in range(len(tfidf)):
#     s = "Cuisine " + CUISINES[i] + " TFIDF:"

#     for j in range(len(tfidf[i])):
#         s += " (" + dict.get(tfidf[i][j][0]) + ","
#         s = s + ("%.3f" % tfidf[i][j][1]) + ")"

#     print(s)
#     print()

# Print Similarity matrix
for i in range(len(corp)):
    print(CUISINES[i] + " Cuisine:")

    sim = index[tfidf_model[corp[i]]]
    for j in range(len(sim)):
        print("Similarity with " + CUISINES[j] + " cuisine " + ("%.3f " % sim[j]))

    print()
