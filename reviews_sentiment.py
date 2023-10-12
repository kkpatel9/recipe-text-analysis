import pandas as pd
import sys
import os
import re
from statistics import mean

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Import the module from the parent directory
from sentiment_module import sentiment

# Cuisines list
CUISINES = [
    "Mexican", "Italian", "Indian", "Thai", "Korean", "French", "Latin American", "Chinese", "Japanese", "Spanish"
]

# Initialize term vector
term_vec = []

# Make data frame for review and arousal/valence
df = pd.DataFrame(columns = ["review_sentence", "arousal", "valence", "cuisine"]) 

# Loop through all cuisines
for cuisine in CUISINES:
    # Cuisine progress
    print("Cuisine: " + cuisine)

    # Read reviews CSV
    df_instr = pd.read_csv(f"data/reviews/individual/{cuisine}_reviews.csv")

    reviews = ""
    arousals = []
    valences = []

    # Loop through all reviews
    for i, row in df_instr.iterrows():
        # Extract review
        review = row["Review"]

        # Replace the redundant words
        review = review.replace("icons", "").replace("ellipsis-horizontal", "").replace("ellipsis", "").replace("reply", "")

        # Append to to reviews
        reviews += review

        # Split up review by sentence
        review_sentences = re.split(r"[.;!?-]", review)

        # Loop through review sentences
        for sentence in review_sentences:
            sentence_terms = sentence.split()

            # Get sentiment of the sentence
            review_sentiment = sentiment.sentiment(sentence_terms)

            # If sentiment values are 0 for both, could signify blank space or names of writers
            if review_sentiment["arousal"] != 0 or review_sentiment["valence"] != 0:
                arousals.append(review_sentiment["arousal"])
                valences.append(review_sentiment["valence"])

                # Append to dataframe
                df.loc[len(df.index)] = [sentence, review_sentiment["arousal"], review_sentiment["valence"], cuisine] 

    # Average the arousal and valences from all reviews
    cuisine_arousal = mean(arousals)
    cuisine_valence = mean(valences)

    # Output stats
    print("Average Review Arousal: " + "%.3f " % cuisine_arousal)
    print("Average Review Valence: " + "%.3f " % cuisine_valence)
    print()

# Export dataframe
df.to_csv("data/reviews/review_sentiment.csv", index = False)