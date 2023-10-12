from bs4 import BeautifulSoup
import requests
import string
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords

# Dict for all cuisines
CUISINE_LINKS = {
    "Mexican": "https://www.food.com/ideas/mexican-food-at-home-6830?ref=nav#c-706013",
    "Italian": "https://www.food.com/ideas/italian-food-recipes-at-home-6828?ref=nav",
    "Indian": "https://www.food.com/ideas/indian-food-recipes-at-home-6821?ref=nav",
    "Thai": "https://www.food.com/ideas/thai-food-recipes-at-home-6820?ref=nav",
    "Korean": "https://www.food.com/ideas/korean-food-recipes-at-home-7143?ref=nav",
    "French": "https://www.food.com/ideas/french-food-at-home-7129?ref=nav",
    "Latin American": "https://www.food.com/ideas/best-latin-american-recipes-7133?ref=nav",
    "Chinese": "https://www.food.com/ideas/chinese-food-at-home-6807?ref=nav",
    "Japanese": "https://www.food.com/ideas/japanese-food-recipes-at-home-7140?ref=nav",
    "Spanish": "https://www.food.com/ideas/spanish-food-recipes-at-home-7122?ref=nav"
}

# Grab stop words
stop_words = list(stopwords.words("english"))

# When collecting ingredients, we want to remove stop words plus common recipe terms
REDUNDANT_WORDS = stop_words + [ "cup", "cups", "teaspoon", "teaspoons", "tablespoon", "tablespoons", "lb", "optional",
                                "garnish", "ml", "taste", "ground", "fresh", "sliced", "minced", "seeded", "freshly", "cut",
                                 "chopped", "large", "dried", "lbs", "ounce", "ounces", "g", "finely", "thinly", "grated", "dry" ]

# Final data frame to append to
final_ingredients_df = pd.DataFrame(columns = ["ingredient", "count", "cuisine"])

# Loop through all cuisines
for cuisine in CUISINE_LINKS:
    print("Processing cuisine: " + cuisine)

    # Get response
    result = requests.get(CUISINE_LINKS[cuisine])

    # Get text
    content = result.text

    # Run through beautfiul soup
    soup = BeautifulSoup(content, "lxml")

    # Initialize ingredients list
    ingredients = []

    # Find all recipe links
    for link in soup.find_all("a"):
        # Pull href link from it
        link_string = link.get("href")

        # Find /recipe/
        if link_string and "www.food.com/recipe/" in link_string:

            # Response for recipe
            recipe_result = requests.get(link_string)

            # Pull recipe content
            recipe_content = recipe_result.text
            soup = BeautifulSoup(recipe_content, "lxml")

            # Get all ingredients
            ingredients_html = soup.find_all("span", class_="ingredient-text svelte-1dqq0pw")
            
            for x in ingredients_html:
                # Get each ingredient
                ingredient = x.get_text(strip = True, separator = ' ').lower().translate(str.maketrans('', '', string.punctuation))

                # Process the ingredient to not include numbers or extra words
                pre_ingredient = ingredient.split()
                ingredient_tmp = []

                for x in pre_ingredient:
                    # Check if there are any numbers in the string, we don't want them
                    if not any(char.isdigit() for char in x):
                        # Remove any common ingredient terms that is not meaningful
                        if x not in REDUNDANT_WORDS:
                            ingredient_tmp.append(x)

                # Add to list
                ingredients += ingredient_tmp

    # Count occurrences of ingredients
    count = Counter(ingredients)

    # Convert to dataframe
    df = pd.DataFrame.from_dict(count, orient="index", columns=["count"])
    df["ingredient"] = df.index

    # Sort by descending count
    df = df.sort_values(by=["count"], ascending=False)

    # Add cuisine and append to main dataframe
    df["cuisine"] = cuisine.lower()
    final_ingredients_df = pd.concat([final_ingredients_df, df])

# Export
final_ingredients_df.to_csv("data/ingredient_counts/ingredients_by_cuisine.csv", index=False)
