import pandas as pd
from nltk.corpus import stopwords
import re
from collections import Counter

# Cuisines list
CUISINES = [
    "Mexican", "Italian", "Indian", "Thai", "Korean", "French", "Latin American", "Chinese", "Japanese", "Spanish"
]

# Grab stop words
stop_words = list(stopwords.words("english"))

final_reviews_df = pd.DataFrame(columns = ["review_term", "count", "cuisine"])

for cuisine in CUISINES:
    # Read reviews CSV
    df_instr = pd.read_csv(f"data/reviews/individual/{cuisine}_reviews.csv")
    reviews = ""

    for i, row in df_instr.iterrows():
        # Make review lowercase
        review = row["Review"].lower()

        # Replace the redundant words
        review = review.replace("icons", "").replace("ellipsis-horizontal", "").replace("ellipsis", "").replace("reply", "")

        # Replace punctuation
        review = re.sub( r"[^\w\s]", "", review )

        reviews += review

    # Split by space
    reviews_terms_raw = reviews.split()

    # Remove stop words
    intructions_terms = []

    for x in reviews_terms_raw:
        # Check if there are any numbers in the string, we don't want them
        if not any(char.isdigit() for char in x):
            # Remove any common ingredient terms that is not meaningful
            if x not in stop_words:
                intructions_terms.append(x)

    # Count occurrences of terms
    count = Counter(intructions_terms)

    # Convert to dataframe
    count_df = pd.DataFrame.from_dict(count, orient="index", columns=["count"])
    count_df["review_term"] = count_df.index

    # Sort
    count_df = count_df.sort_values(by=["count"], ascending=False)

    # Add cuisine and append to main dataframe
    count_df["cuisine"] = cuisine.lower()
    final_reviews_df = pd.concat([final_reviews_df, count_df])

# Export
final_reviews_df.to_csv("data/reviews/reviews_by_cuisine.csv", index=False)