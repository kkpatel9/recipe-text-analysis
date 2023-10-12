import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
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
    # Read reviews CSV
    df_instr = pd.read_csv(f"data/reviews/individual/{cuisine}_reviews.csv")
    reviews = ""

    for i, row in df_instr.iterrows():
        # Extract review
        review = row["Review"]

        # Replace the redundant word 'directions'
        review = review.replace("icons", "").replace("ellipsis-horizontal", "").replace("ellipsis", "").replace("reply", "")

        # Append to reviews
        reviews += review

    # Remove punctuation, then tokenize documents
    punc = re.compile( '[%s]' % re.escape(string.punctuation))
    reviews = reviews.lower()
    reviews = punc.sub("", reviews)

    # Add to term vector
    term_vec.append(nltk.word_tokenize(reviews))

# Porter stem remaining terms
porter = nltk.stem.porter.PorterStemmer()

for i in range(len(term_vec)):
    for j in range(len(term_vec[i])):
        term_vec[i][j] = porter.stem(term_vec[i][j])

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

# Create pairwise cuisine similarity index
n = len(dict)
index = gensim.similarities.SparseMatrixSimilarity(tfidf_model[corp], num_features = n)

# Print Similarity matrix
for i in range(len(corp)):
    print(CUISINES[i] + " Cuisine:")

    sim = index[tfidf_model[corp[i]]]
    for j in range(len(sim)):
        print("Similarity with " + CUISINES[j] + " cuisine " + ("%.3f " % sim[j]))

    print()