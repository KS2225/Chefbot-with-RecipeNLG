import warnings
import numpy as np
warnings.filterwarnings('ignore')
import pandas as pd
from nltk.tokenize import word_tokenize
import json
import random
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from json.decoder import JSONDecodeError
greetings = pd.read_csv("Greetings.csv")

def main_menu(bot_name, user_name,vectorizer):                                                              #print food menu
    while True:
        print(f"\n{bot_name}: Pick a following category- \n\t1. Soup\n\t2. Non-vegetarian meals\n\t3. Vegan meals\n\t4. Dessert\n\t (Enter corresponding number or '0' to go back)")
        try:
            menu_input = int(input(f"\n{user_name}: "))
            if menu_input == 0:
                print("Welcome back to the main menu")
                return False
            elif menu_input in range(1, 5):
                sub_menu(menu_input, bot_name, user_name,vectorizer)
                break  # Exit the loop if sub_menu is executed successfully
            else:
                print(f"\n{bot_name}: Invalid input. Please enter a number between 0 and 4.")
        except ValueError:
            print(f"\n{bot_name}: Invalid input. Please enter a number.")

def sub_menu(index, bot_name, user_name,vectorizer):    #assign csv files and delve into sub-categories
    if index == 1:
        recipe_df = r"Soup.csv"
        direct_cooking(bot_name, recipe_df, user_name, vectorizer)
    if index == 2:
        print(f"\n{bot_name}:Pick type of meat- \n\t1. Chicken\n\t2. Beef\n\t3. Pork")
        ingredient_input = int(input(f"\n{user_name}: "))
        if ingredient_input == 1:
            recipe_df = r"Chicken.csv"
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        elif ingredient_input == 2:
            recipe_df = r"Beef.csv"
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        elif ingredient_input == 3:
            recipe_df = r"Pork.csv" 
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        else:
            main_menu(bot_name, user_name)
    if index == 3:
        recipe_df = r"Vegan.csv"
        direct_cooking(bot_name, recipe_df, user_name, vectorizer)
    if index == 4:
        print(f"\n{bot_name}:Pick type of sweet- \n\t1. Cake\n\t2. Candy\n\t3. Biscuits\n\t4. Cookies\n\t5. Others")
        ingredient_input = int(input(f"\n{user_name}: "))
        if ingredient_input == 1:
            recipe_df = r"Cake.csv"
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        elif ingredient_input == 2:
            recipe_df = r"Candy.csv"
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        elif ingredient_input == 3:
            recipe_df = r"Biscuit.csv" 
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        if ingredient_input == 4:
            recipe_df = r"Cookie.csv"
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        elif ingredient_input == 5:
            recipe_df = r"Dessert.csv"
            direct_cooking(bot_name, recipe_df, user_name, vectorizer)
        else:
            main_menu(bot_name, user_name,vectorizer)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()                                        # convert sentences into tokens
    words = word_tokenize(text)                                             # Lemmatize tokens
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]       
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

def preprocess_directions(directions_data):                                   # Remove unnecessary characters from directions
    cleaned_data = re.sub(r'\["|"\]', '', directions_data)
    cleaned_data = re.sub(r'[.]\s*,?\s*', '. ', cleaned_data)
    cleaned_data = re.sub(r'[.]\s*\n\s*', '. ', cleaned_data)
    cleaned_data = cleaned_data.replace('"', '').replace(',', '')
    return cleaned_data

def retrieve_directions(recipe_file_path, recipe_name):
    try:
        recipe_df = pd.read_csv(recipe_file_path)
        recipe_entry = recipe_df[recipe_df['title'].str.lower() == recipe_name.lower()]
        if not recipe_entry.empty:
            if 'directions' in recipe_entry and recipe_entry['directions'].iloc[0]:                     # Check if 'directions' key exists
                directions_data = preprocess_directions(recipe_entry['directions'].iloc[0])             # Preprocess the directions data
                directions = nltk.sent_tokenize(directions_data)                                        # Split the directions into a list of steps
                return directions
            else:                                                                                       #Handle  errors
                print("I'm sorry but I can't fetch the directions right now! Maybe try again later")
                return None
    except JSONDecodeError as json_error:                                                               #Handle  errors 
        print(f"I'm sorry but I can't fetch the directions right now! Maybe try again later")
        return None
    except Exception as e:
        print(f"I'm sorry but I can't fetch the directions right now! Maybe try again later")
        return None  

def retrieve_ingredients(recipe_file_path, recipe_name):
    try:
        recipe_df = pd.read_csv(recipe_file_path)
        recipe_entry = recipe_df[recipe_df['title'].str.lower() == recipe_name.lower()]
        if 'ingredients' in recipe_entry and recipe_entry['ingredients'].iloc[0]:                           # Check if 'ingredients' key exists
            ingredients_data = recipe_entry['ingredients'].iloc[0].replace('\'', '"')                       # load JSON data
            ingredients = json.loads(ingredients_data)
            return ingredients
        else:                                                                                               # Handle the errors
            print("I'm sorry but I can't fetch the ingredients right now! Maybe try again later")
            return None 
    except JSONDecodeError as json_error:                                                                   # Handle the errors
        print("I'm sorry but I can't fetch the ingredients right now! Maybe try again later")
        return None
    except Exception as e:
        print("I'm sorry but I can't fetch the ingredients right now! Maybe try again later")
        return None 

# Greetings intent matching and responses
def match_intent(user_input,vectorizer):
    user_input = lemmatize_text(user_input)
    greetings['Question'] = greetings['Question'].str.lower()
    intent_vectors = vectorizer.fit_transform(greetings['Question'])
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input])                                                        # Vectorize input
    similarities = cosine_similarity(user_vector, intent_vectors).flatten()                                 # cosine similarity
    highest_similarity_indices = [i for i, sim in enumerate(similarities) if sim == max(similarities)]      # Identify the index of the best match
    similarity_threshold = 0.7
    if max(similarities) > similarity_threshold:                                                            # if similarity score > 0.7, choose intent
        best_match_index = random.choice(highest_similarity_indices)
        best_match_intent = greetings.iloc[best_match_index].to_dict()
        return best_match_intent
    else:
        return False

# QnA intent matching and responses
def smalltalk_answer(questions_df, query, vectorizer,bot_name):
    query = lemmatize_text(query)
    questions_df['Question'] = questions_df['Question'].str.lower().apply(lemmatize_text)                   # Vectorize csv
    intent_vectors = vectorizer.fit_transform(questions_df['Question'])
    query = query.lower()
    user_vector = vectorizer.transform([query])                                                             # Vectorize input
    similarities = cosine_similarity(user_vector, intent_vectors).flatten()                                 # cosine similarity
    highest_similarity_indices = [i for i, sim in enumerate(similarities) if sim == max(similarities)]      # Identify the index of the best match
    similarity_threshold = 0.6
    if max(similarities) > similarity_threshold:                                                            # if similarity score > 0.6, choose intent
        best_match_index = random.choice(highest_similarity_indices)
        best_match_intent = questions_df['Answer'].iloc[best_match_index]
        return f"\n{bot_name}: {best_match_intent}"
    else:
        return False

def recipe_list(recipe_file_path, user_input, vectorizer, excluded_recipe=None):
    recipe_df = pd.read_csv(recipe_file_path)
    recipe_df['combined_text'] = recipe_df['title'] + ' ' + recipe_df['ingredients']        # Combine columns for vectorization
    vectorized_text = vectorizer.fit_transform(recipe_df['combined_text'])                  # Vectorize the combined_text column
    user_input_vector = vectorizer.transform([user_input])                                  # Vectorize the user input
    similarity_scores = cosine_similarity(vectorized_text, user_input_vector).flatten()     # Calculate the similarity scores
    recipe_df['similarity'] = similarity_scores                                             # Add the similarity scores to the DataFrame
    sorted_recipes = recipe_df.sort_values(by='similarity', ascending=False)                # Sort by similarity in descending order
    if excluded_recipe is not None:
        sorted_recipes = sorted_recipes[sorted_recipes['title'] != excluded_recipe]         # Filter out the excluded recipe
    recommended_recipe = sorted_recipes.iloc[0]['title']                                    # Get the top recommended recipe
    return recommended_recipe

def ingredient_suggestion(recipe_file_path, user_input, bot_name,vectorizer):
    recipes_df = pd.read_csv(recipe_file_path)
    query = lemmatize_text(user_input)
    recipes_df['NER'] = recipes_df['NER'].str.lower()                                                               #NER column has a list of main ingredients for each recipe
    recipes_df['NER'] = recipes_df['NER'].apply(lemmatize_text)
    intent_vectors = vectorizer.fit_transform(recipes_df['NER'])                                                    # Fit the vectorizer on the training data
    query = query.lower()
    user_vector = vectorizer.transform([query])
    similarities = cosine_similarity(user_vector, intent_vectors).flatten()                                         # Calculate cosine similarity between the user's input and each recipe vector
    best_match_index = np.argmax(similarities)                                                                      # Identify the index of the best match
    similarity_threshold = 0.5
    if similarities[best_match_index] > similarity_threshold:
        best_match_recipe = recipes_df['NER'].iloc[best_match_index]
        recipe_entry = recipes_df[recipes_df['NER'].str.lower() == best_match_recipe]
        if not recipe_entry.empty:
            name = recipe_entry['title'].iloc[0].replace('\'', '"')                                                 # Retrieve ingredients
            return name
        else:
            return f"\n{bot_name}: I couldn't find a matching recipe."                                              # if no match, print error message
    else:
        return f"\n{bot_name}: I couldn't find a matching recipe."

def cooking(bot_name, user_name,vectorizer):                                       # function to handle the use case where user is redirected towards cooking intent
    file_list = ["Beef", "Chicken", "Biscuit", "Cake", "Candy", "Cookie", "Dessert", "Pork", "Vegan", "Pizza", "Spaghetti", "Burger", "Rice"]
    print(f"\n{bot_name}: Let me fetch my cookbook!")
    print(f"\n{bot_name}: Do you have a recipe in mind?(yes/no)")                  # ask if user knows what to cook
    u_in = input(f"\n{user_name}: ") 
    recipe_df = None  # Assign a default value              
    if u_in.lower() == "yes":                                                      # if yes,
        print(f"\n{bot_name}: Enter your recipe name: ")                            # take recipe name
        rec_name = input(f"\n{user_name}: ")
        for word in file_list:                                                      # for generic names, find name in file_list and set it as recipe_df
            if rec_name.lower() == word.lower():
                recipe_df = f"{word}.csv"
                break
            regex_pattern = re.compile(word, re.IGNORECASE)
            if re.search(regex_pattern, rec_name):
                recipe_df = f"{word}.csv"
                break      
        if recipe_df is not None:
            print(f"\n{bot_name}: Hmm! Let me check...")
            find_recipe(recipe_df, rec_name, bot_name, user_name,vectorizer)
        else:
            print(f"\n{bot_name}: I couldn't find the recipe. Please be more precise")          #if recipe not found, try again
            cooking(bot_name,user_name,vectorizer)  
    elif u_in.lower() == "no":
        print(f"\n{bot_name}: Here's a menu You can go through!")                     # if user doesn't know what to cook, redirect to food menu
        main_menu(bot_name, user_name,vectorizer)
    elif u_in.lower() == "exit":
        print("Welcome back to the main menu")
        return
    else:
        print(f"\n{bot_name}: I'm sorry, I didn't understand that.")                # handle ambiguous input
        cooking(bot_name,user_name,vectorizer)  

def find_recipe(recipe_df, rec_name, bot_name, user_name, vectorizer):              #layout to confirm recipe name
    while True:
        most_similar_recipe = recipe_list(recipe_df, rec_name, vectorizer, excluded_recipe=None)
        if most_similar_recipe:
            print(f"\n{bot_name}: Do you mean {most_similar_recipe}?")
            u_in2 = input(f"\n{user_name}: ")
            if u_in2.lower() == "yes":
                print(f"\n{bot_name}: Let's get cooking {user_name}!")
                print("\t Collect the following ingredients: ")
                if proceeding(recipe_df, rec_name, bot_name, user_name, vectorizer):
                    break                                                           # Exit the loop after cooking is done
                return
            elif u_in2.lower() == "exit":
                print("Welcome back to the main menu")
                return
            else:
                next_best_recipe = recipe_list(recipe_df, most_similar_recipe, vectorizer, excluded_recipe=most_similar_recipe)
                print(f"\n{bot_name}: How about: {next_best_recipe}")
                u_in5 = input(f"\n{user_name}: ")
                if u_in5.lower() == "yes" or u_in5.lower() == "ok":
                    print(f"\n{bot_name}: Let's get cooking {user_name}!")
                    print("\t Collect the following ingredients: ")
                    if proceeding(recipe_df, next_best_recipe, bot_name, user_name, vectorizer):
                        break  # Exit the loop after cooking is done
                else:
                    print(f"\n{bot_name}: Let's try that again but this time with a little more detail")
                    cooking(bot_name, user_name, vectorizer)
                    break  # Exit the loop if cooking function is called
            return
        else:
            print(f"\n{bot_name}: I couldn't find a matching recipe.")
            break  # Exit the loop if no matching recipe is found

def direct_cooking(bot_name, recipe_file_path, user_name, vectorizer):
    count = 0
    while count < 4:
        recipe_df = pd.read_csv(recipe_file_path)
        random_recipe = recipe_df.sample(n=1).iloc[0]['title']  # Randomly select a recipe from the DataFrame
        print(f"\n{bot_name}: Here is a recommendation: ")
        print(f"\t How about trying {random_recipe}?")
        u_in = input(f"\n{user_name}: ").lower()
        if u_in in {"yes", "ok"}:
            print(f"\n{bot_name}: Let's get cooking {user_name}!")
            ingredients = retrieve_ingredients(recipe_file_path, random_recipe)                                 # Retrieve ingredients
            if ingredients is not None:
                print("\t Collect the following ingredients: ")
                for idx, ingredient in enumerate(ingredients, start=1):
                    print(f"\t {idx}: {ingredient}")
                print(f"\n{bot_name}: Do you want to proceed with making {random_recipe}?")                     # confirm if user has ingredients
                u_in4 = input(f"\n{user_name}: ").lower()
                if u_in4 in {"yes", "ok"}:
                    result = final_cooking(bot_name, recipe_file_path, random_recipe, user_name,vectorizer)     # if ingredients are available, proceed to cooking
                    if result == 'exit':
                        print("Welcome back to the main menu")
                        return                                                                                  # Exit the function and return to the main menu
                elif u_in4 in {"exit"}:
                    if result == 'exit':
                        print("Welcome back to the main menu")
                        return                                                                                  # Exit the function and return to the main menu
                else:
                    print(f"\n{bot_name}: Okay, let me suggest another recipe.")                                # if user doesn't have ingredients, suggest another recipe
            else:
                print(f"\n{bot_name}: Unable to retrieve ingredients.")                                         # If parsing error, print error message
            return   
        elif u_in == "no":
            count += 1
            print(f"\n{bot_name}: How about trying a different recipe?")                                        # if user doesn't like suggestion, suggest another recipe
        elif u_in == "exit":
            print(f"\n{bot_name}: Closing Cookbook")
            print("Welcome back to the main menu")
            return                                                                                              # Exit the function and return to the main menu  
        else:
            print(f"\n{bot_name}: Let's start from the top ")                                                   # for no viable option, redirect user to food menu
            main_menu(bot_name, user_name,vectorizer)
            break
    if count >= 3:                                                            # if user doesn't like the first 3 recommendations, ask user for ingredients they have
        print(f"\n{bot_name}: Ok! Why don't you tell me what ingredients you have, {user_name}, so I can recommend a dish?\n\t (yes/no)")
        u_in2 = input(f"\n{user_name}: ").lower()

        if u_in2 in {"yes", "ok"}:
            print(f"\n{bot_name}: Enter your main ingredients: ")                                               # take ingredient names as input
            u_in3 = input(f"\n{user_name}: ")
            recipe_suggested = ingredient_suggestion(recipe_file_path, u_in3, bot_name,vectorizer)
            print(f"\n{bot_name}: This is what you can make: {recipe_suggested}")                               # suggest a recipe using ingredient_suggestion function
            
            ingredients_needed = retrieve_ingredients(recipe_file_path, recipe_suggested)
            
            if ingredients_needed is not None:
                print(f"\n{bot_name}: Here is what you will need:")                                             # print ingredients
                for idx, ingredient in enumerate(ingredients_needed, start=1):
                    print(f"\t {idx}: {ingredient}")
                    
                print(f"\n{bot_name}: Do you want to proceed with making {recipe_suggested}?")                  #confirm recipe
                u_in4 = input(f"\n{user_name}: ").lower()

                if u_in4 in {"yes", "ok"}:
                    result = final_cooking(bot_name, recipe_file_path, random_recipe, user_name,vectorizer)     # move to the cooking stage

                    if result == 'exit':
                        print("Welcome back to the main menu")
                        return                                                                                  # Exit the function and return to the main menu
                else:                                                                                           # else suggest next best recipe match
                    next_best_recipe = recipe_list(recipe_file_path, recipe_suggested, vectorizer, excluded_recipe=recipe_suggested)
                    print(f"\n{bot_name}: How about: {next_best_recipe}?")
                    u_in5 = input(f"\n{user_name}: ").lower()

                    if u_in5 in {"yes", "ok"}:
                        ingredients = retrieve_ingredients(recipe_df, next_best_recipe)

                        if ingredients is not None:
                            print("\t Collect the following ingredients: ")
                            for idx, ingredient in enumerate(ingredients, start=1):                            # print ingredients
                                print(f"\t {idx}: {ingredient}")

                            print(f"\n{bot_name}: Do you want to proceed with making {next_best_recipe}?")     #confirm recipe
                            u_in4 = input(f"\n{user_name}: ").lower()

                            if u_in4 in {"yes", "ok"}:
                                final_cooking(bot_name, recipe_df, next_best_recipe, user_name)                # move to the cooking stage
                                # print("Welcome back to the main menu")
                                return  # Exit the function after final_cooking

                            else:
                                print(f"\n{bot_name}: How about you go through the menu again?")                #redirect to food menu
                                main_menu(bot_name, user_name)
                                return
                        else:
                            print(f"\n{bot_name}: Unable to retrieve ingredients.")
                    else:
                        print(f"\n{bot_name}: How about one final recommendation? ")                           # else suggest one final recipe 
                        next_best_recipe = recipe_list(recipe_file_path, next_best_recipe, vectorizer, excluded_recipe=next_best_recipe)
                        print(f"\n{bot_name}: Would you like to make {next_best_recipe} ?")
                        u_in5 = input(f"\n{user_name}: ").lower()

                        if u_in5 in {"yes", "ok"}:
                            final_cooking(bot_name, recipe_file_path, recipe_suggested, user_name)
                            return
                        else:
                            print(f"\n{bot_name}: How about you go through the menu again?")                    #redirect to food menu
                            main_menu(bot_name, user_name, vectorizer)
                            return
            else:
                print(f"\n{bot_name}: Unable to retrieve ingredients.")
        else:
            print(f"\n{bot_name}: How about you go through the menu again?")                                   #redirect to food menu
            main_menu(bot_name, user_name,vectorizer)
            return

def proceeding(recipe_file_path, rec_name, bot_name, user_name,vectorizer):         #fetch ingredients and confirm choice
    ingredients = retrieve_ingredients(recipe_file_path, rec_name)
    if ingredients is not None:
        for idx, ingredient in enumerate(ingredients, start=1):
            print(f"\t {idx}: {ingredient}")
        print(f"\n{bot_name}: Do you want to proceed with making {rec_name}?")   
        u_in4 = input(f"\n{user_name}: ")
        if u_in4.lower() == "yes" or u_in4.lower() == "ok":
            final_cooking(bot_name, recipe_file_path, rec_name, user_name,vectorizer)       # proceed to cooking stage
            return  # Exit the function after final_cooking
    else:
        print(f"\n{bot_name}: Unable to retrieve ingredients.")

def final_cooking(bot_name, recipe_file_path, recipe_name, user_name, vectorizer):                              # guide user through recipe steps
    current_step = 0
    print(f"\n{bot_name}: Fantastic choice! Let's start cooking.")
    directions = retrieve_directions(recipe_file_path, recipe_name)                                            # retrieve directions from the csv file using retrieve_directions function
    while current_step < len(directions):
        step = directions[current_step]
        print(f"\n{bot_name}: Step {current_step + 1}: {step}")     
        user_input = input(f"\n{bot_name}: Type 'next' to proceed, 'back' to go back and 'exit' to stop cooking: ")
        if user_input.lower() == 'next':                                                                        # if user input is "next", go to next step
            current_step += 1
            if current_step == len(directions) - 2:
                print(f"\n{bot_name}: You're almost there, {user_name}! Just a few more steps to go.")          #conversational marker
            elif current_step == len(directions)//2:
                print(f"\n{bot_name}: You've reached halfway! Great job so far. If you're ready to continue, type 'next'.")     #conversational marker
        elif user_input.lower() == 'back':                                                                     # if user input is "back", go to previous step
            if current_step > 0:
                current_step -= 1
                print(f"\n{bot_name}: Sure, let's go back to Step {current_step + 1}.")
            else:
                print(f"\n{bot_name}: We're already at the first step.")                                        # user can't go back from step 1
        elif '?' in user_input:
            handle_user_question(recipe_file_path, recipe_name, bot_name)                                       # if user asks a question, redirect to handle_user_question function
        elif user_input.lower() == 'exit':
            print(f"\n{bot_name}: Returning to the main menu.")                                                 #go back to main menu
            # print("Welcome back to the main menu")
            return "exit"
        else:
            print(f"\n{bot_name}: I'm sorry, I didn't understand. Please type 'next' to proceed, 'back' to go back, or ask a question.") # handle ambiguous input
    print(f"\n{bot_name}: You've completed all the steps. Enjoy your dish!\n{bot_name}: Returning to the main menu.")                    # cooking complete marker
    print("Welcome back to the main menu")
    return
    
def handle_user_question(recipe_file_path, recipe_name,bot_name):
    recipe_df = pd.read_csv(recipe_file_path)
    try:
        recipe_entry = recipe_df[recipe_df['title'].str.lower() == recipe_name.lower()]                             # Locate the recipe name based on user input
        if not recipe_entry.empty and 'link' in recipe_entry and pd.notna(recipe_entry['link'].iloc[0]):            # Check if 'link' key exists and has a non-empty value
            link = recipe_entry['link'].iloc[0]                                                                     # Extract the link string
            print(f"\n{bot_name}: For further details, visit:- {link}")
            return link
        else:                                                                                                       # Handle errors
            print(f"{bot_name}: Cannot find the link.")
            return None 
    except Exception as e:
        print(f"{bot_name}: Cannot find the link.")       
        return

def main():
    # Read the recipe dataframes
    recipe_dfs = {}
    for word in ["Beef", "Chicken", "Biscuit", "Cake", "Candy", "Cookie", "Dessert", "Pork", "Vegan", "Pizza", "Spaghetti", "Burger", "Rice"]:
        recipe_dfs[word] = pd.read_csv(f"{word}.csv")
    questions_df = pd.read_csv("COMP3074-CW1-Dataset.csv")                   # Read the questions dataframe
    vectorizer = TfidfVectorizer()                                                          # Vectorizer for QnA and recipes
    questions_df['Question'] = questions_df['Question'].str.lower()
    questions_df['Question'] = questions_df['Question'].apply(lemmatize_text)
    print("Waking up Chefbot...")
    print("")
    bot_name = "Chefbot"
    print(f"\n{bot_name}: Hello! My name is Chefbot!")
    print(f"\n{bot_name}: To start cooking, just type 'I want to cook'")
    print("What may I refer to you as?")
    user_name = input("You: ")                                                              #take user name as input
    user_name = user_name.capitalize()
    if user_name.lower() == "exit":                                                         #quit code if "exit"
        print(f"\n{bot_name}: Goodbye!")
        quit()
    print(f"\n{bot_name}: Hello {user_name}!")
    while True:
        user_input = input(f"\n{user_name}: ")
        if user_input.lower() == "bye":
            print(f"\n{bot_name}: Goodbye!")
            quit()
        matched_intent_data = match_intent(user_input,vectorizer)
        if matched_intent_data:
            if "goodbye" in matched_intent_data["Class"]:                                           # if "goodbye" then exit
                    print(f"\n{bot_name}: {(matched_intent_data['Answer'])}")
                    quit()
            if "name_question" in matched_intent_data["Class"]:                                     # if identity question
                    print(f"\n{bot_name}:{matched_intent_data['Answer'].format(user_name)}.")  
            else:
                answer = match_intent(user_input,vectorizer)['Answer']                          # match other greeting intents
                print(f"\n{bot_name}:",answer)
        elif user_input.lower() == "i want to cook":                                            # cooking intent
            cooking(bot_name, user_name,vectorizer)
        else:
            best_match_intent = smalltalk_answer(questions_df, user_input, vectorizer,bot_name)    #small talk QnA intent
            if best_match_intent:
                print(best_match_intent)
            else:
                print(f"\n{bot_name}: I'm sorry, I didn't understand that.")                        #ambiguous input

if __name__ == "__main__":
    main()