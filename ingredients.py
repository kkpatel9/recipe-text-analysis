from bs4 import BeautifulSoup
import requests
import pandas as pd

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

# Loop through all cuisines
for cuisine in CUISINE_LINKS:
    print("Processing cuisine: " + cuisine)

    # Get response
    result = requests.get(CUISINE_LINKS[cuisine])

    # Get text
    content = result.text

    # Run through beautfiul soup
    soup = BeautifulSoup(content, "lxml")

    # Initialize ingredients string
    ingredients = ""

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
                # Add each ingredient
                ingredients += x.get_text(strip = True, separator = ' ')

    # Make and export dataframe as CSV
    df = pd.DataFrame({"ingredients": [ingredients]})
    df.to_csv(f"data/ingredients/{cuisine}_ingredients.csv")