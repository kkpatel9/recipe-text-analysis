import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Cuisines list
CUISINES = [
    "Mexican", "Italian", "Indian", "Thai", "Korean", "French", "Latin American", "Chinese", "Japanese", "Spanish"
]

# Create inital docs list
docs = []

def display_doc_2_topic(doc_2_topic, collect):
    for i in range(0, len(collect)):
        topic_wt = list(doc_2_topic[i])
        idx = topic_wt.index(max(topic_wt))

        print(collect[i] + ":")
        print(f"  Concept {idx}, {topic_wt[ idx ] * 100.0:.02f}%")

def display_topics(model, feat_nm, top_word_n):
    for i, topic in enumerate(model.components_):
        print(f"Concept {i}:")
        topic_len = sum(topic)

        term = " ".join(
            [
                f"{feat_nm[i]} ({topic[i] / topic_len * 100.0:.02f}%); "
                for i in topic.argsort()[: -top_word_n - 1 : -1]
            ]
        )
        print("   " + term)

# Loop through all cuisines
for cuisine in CUISINES:
    # Read instructions CSV
    df_ing = pd.read_csv(f"data/ingredients/{cuisine}_ingredients.csv")
    docs.append(df_ing.iloc[0,1])

feat_n = 10

#  Raw term counts for LDA
tf_vectorizer = CountVectorizer(
    max_df=0.95, min_df=2, max_features=feat_n, stop_words="english"
)
tf = tf_vectorizer.fit_transform(docs)
tf_feat_nm = tf_vectorizer.get_feature_names_out()

topic_n = 5
lda = LatentDirichletAllocation(
    n_components=topic_n,
    max_iter=5,
    learning_method="online",
    learning_offset=50.0,
    random_state=0,
)

lda_topic = lda.fit(tf)
doc_2_topic = lda.transform(tf)

top_word_n = 10
display_topics(lda, tf_feat_nm, top_word_n)