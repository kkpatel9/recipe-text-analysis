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

final_instructions_df = pd.DataFrame(columns = ["instruction_term", "count", "cuisine"])

for cuisine in CUISINES:
    # Read instructions CSV
    df_instr = pd.read_csv(f"data/instructions/individual/{cuisine}_instructions.csv")
    instructions = ""

    for i, row in df_instr.iterrows():
        # Make instruction lowercase
        instruction = row["Instructions"].lower()

        # Replace the redundant word 'directions'
        instruction = instruction.replace("directions", "")

        # Replace punctuation
        instruction = re.sub( r"[^\w\s]", "", instruction )

        instructions += instruction

    # Split by space
    instructions_terms_raw = instructions.split()

    # Remove stop words
    intructions_terms = []

    for x in instructions_terms_raw:
        # Check if there are any numbers in the string, we don't want them
        if not any(char.isdigit() for char in x):
            # Remove any common ingredient terms that is not meaningful
            if x not in stop_words:
                intructions_terms.append(x)

    # Count occurrences of terms
    count = Counter(intructions_terms)

    # Convert to dataframe
    count_df = pd.DataFrame.from_dict(count, orient="index", columns=["count"])
    count_df["instruction_term"] = count_df.index

    # Sort
    count_df = count_df.sort_values(by=["count"], ascending=False)

    # Add cuisine and append to main dataframe
    count_df["cuisine"] = cuisine.lower()
    final_instructions_df = pd.concat([final_instructions_df, count_df])

# Export
final_instructions_df.to_csv("data/instructions/instructions_by_cuisine.csv", index=False)