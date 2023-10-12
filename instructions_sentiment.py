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

# Make data frame for instruction and arousal/valence
df = pd.DataFrame(columns = ["instruction_sentence", "arousal", "valence", "cuisine"]) 

# Loop through all cuisines
for cuisine in CUISINES:
    # Cuisine progress
    print("Cuisine: " + cuisine)

    # Read instructions CSV
    df_instr = pd.read_csv(f"data/instructions/individual/{cuisine}_instructions.csv")

    instructions = ""
    arousals = []
    valences = []

    # Loop through all instructions
    for i, row in df_instr.iterrows():
        # Extract instruction
        instruction = row["Instructions"]

        # Replace the redundant word "directions"
        instruction = instruction.replace("directions", "")

        # Append to instructions
        instructions += instruction

        # Split up instruction by sentence
        instruction_sentences = re.split(r"[.;!?-]", instruction)

        # Loop through instruction sentences
        for sentence in instruction_sentences:
            sentence_terms = sentence.split()

            # Get sentiment of the sentence
            instruction_sentiment = sentiment.sentiment(sentence_terms)

            # If sentiment values are 0 for both, could signify blank space or names of writers
            if instruction_sentiment["arousal"] != 0 or instruction_sentiment["valence"] != 0:
                arousals.append(instruction_sentiment["arousal"])
                valences.append(instruction_sentiment["valence"])

                # Append to dataframe
                df.loc[len(df.index)] = [sentence, instruction_sentiment["arousal"], instruction_sentiment["valence"], cuisine] 

    # Average the arousal and valences from all instructions
    cuisine_arousal = mean(arousals)
    cuisine_valence = mean(valences)

    # Output stats
    print("Average Instruction Arousal: " + "%.3f " % cuisine_arousal)
    print("Average Instruction Valence: " + "%.3f " % cuisine_valence)
    print()

# Export dataframe
df.to_csv("data/instructions/instruction_sentiment.csv", index = False)