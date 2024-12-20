import os
import random
import streamlit as st 
from ultralytics import YOLO
import cv2
import time
import numpy as np
import shutil
from gtts import gTTS
import re

from deep_translator import GoogleTranslator
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

@st.cache_resource()    
def main_model():
    model=YOLO('best.pt')

    return model

        
def message():
    
    st.write("""
            📷✨ Please check your image to ensure accuracy. If your vegetable isn't recognized, we're sorry! 🥕🍅 We're working on updates to include more varieties. 
            Thanks for your patience! 🤗
            """)

def upload():
    image=None
    image_filename=None
    initial_image = st.camera_input('Take a picture')
    original_image = initial_image
    temp_path = None
    if initial_image is not None:
        image_filename = f"{int(time.time())}.jpg"
        bytes_data = initial_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
    return image, original_image,image_filename
 
def process_image_with_yolo(pic0):
    names={
                0: 'beet',
                1: 'bell_pepper',
                2: 'cabbage',
                3: 'carrot',
                4: 'cucumber',
                5: 'egg',
                6: 'eggplant',
                7: 'garlic',
                8: 'onion',
                9: 'potato',
                10: 'tomato',
                11: 'zucchini'
            
            }
        
    
    labelslist=[]
    # Load your YOLO model
    
    if pic0 is not None:
        # Perform YOLO prediction on the image
        
        model = main_model()
        
       
        pic0=pic0
        result = model.predict(pic0,conf=0.8, save=True, save_txt=True)
        
        txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))

        if txt_files_exist:
            
            lis = open('runs/detect/predict/labels/image0.txt', 'r').readlines()
            for line in lis:
                bresults=line.split(" ")
                bresults=int(bresults[0])
                clabel=names[bresults]
                labelslist.append(clabel)
                #process_line(line, image_np2)
                
           
        labels_count = {}
        for label in labelslist:
            if label in labels_count:
                labels_count[label] += 1
            else:
                labels_count[label] = 1
        labelslist=[]
        
        try:
            if os.path.exists('runs'):
                shutil.rmtree('runs')
                st.session_state.original_image = None  # Clear the original_image variable
                           
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
              
    
    return labels_count
        
                
                



def generate_recipe_prompt(recipe_count, vegetable_dict):
    prompt = f"""Create {recipe_count} nutritional, delightful and concise Indian cuisine recipes using the following vegetables. Each recipe must follow the exact structure below:

                Recipe Number: Number

                Recipe Name: Name of the Indian dish
                
                Ingredients:
                List of ingredients with quantities
                
                Cooking Instructions:
                1. [First step]
                2. [Second step]
                3. [...]
                
                Nutritional Values(per serving):
                - Calories: [value]
                - Protein: [value]g
                - Carbohydrates: [value]g
                - Fat: [value]g
                - Fiber: [value]g
                
                """
                
    for vegetable, count in vegetable_dict.items():
        prompt += f"- {vegetable} ({count} {'piece' if count == 1 else 'pieces'})\n"
                
    prompt += """
                Note:
                1. Use ONLY the vegetables listed above in your recipes.
                2. Ensure all recipes are authentic Indian dishes or Indian-inspired variations.
                3. Be creative with vegetable combinations while ensuring delicious results suitable for Indian tastes.
                4. Provide clear, concise cooking instructions that include typical Indian cooking methods and spices.
                5. Include accurate nutritional information for each recipe.
                6. Ensure each recipe is unique and different from the others, showcasing the diversity of Indian cuisine.
                7. Strictly adhere to the given structure for each recipe.
                8. Separate the recipes with dashed lines.
                9. Consider regional Indian dishes that use these vegetables.
                10. If possible, mention which region of India the dish is from or inspired by.
                """

    return prompt

def parse_recipes(text):
    recipes = []
    raw_recipes = re.split(r'\n-{3,}\n', text)
    
    for i, raw_recipe in enumerate(raw_recipes, 1):
        recipe = {
            'Recipe Number': f'Recipe {i}',
            'Recipe Name': 'Unnamed Recipe',
            'Ingredients': [],
            'Cooking Instructions': [],
            'Nutritional Values': []
        }
        current_section = None
        
        for line in raw_recipe.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Recipe Number:'):
                recipe['Recipe Number'] = line.split(':', 1)[1].strip()
            elif line.startswith('Recipe Name:'):
                recipe['Recipe Name'] = line.split(':', 1)[1].strip()
            elif line == 'Ingredients:':
                current_section = 'Ingredients'
            elif line == 'Cooking Instructions:':
                current_section = 'Cooking Instructions'
            elif line.startswith('Nutritional Values'):
                current_section = 'Nutritional Values'
            elif current_section:
                if current_section == 'Cooking Instructions':
                    # Remove numbering from instructions
                    line = re.sub(r'^\d+\.\s*', '', line)
                recipe[current_section].append(line)
        
        # Only add the recipe if it has a name, ingredients, and instructions
        if (recipe['Recipe Name'] != 'Unnamed Recipe' and 
            recipe['Ingredients'] and 
            recipe['Cooking Instructions']):
            recipes.append(recipe)
    
    return recipes

def translation(text, target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def generate_recipe(recipe_count, vegetable_dict, target_lang):
    prompt = generate_recipe_prompt(recipe_count, vegetable_dict)
    client = OpenAI(
        base_url='https://api.groq.com/openai/v1',
        api_key=st.secrets['key']
    )
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        
        recipes = parse_recipes(response.choices[0].message.content)
        
        # Translate structural elements
        structure_translations = {
            'Recipe Number': translation('Recipe Number', target_lang),
            'Recipe Name': translation('Recipe Name', target_lang),
            'Ingredients': translation('Ingredients', target_lang),
            'Cooking Instructions': translation('Cooking Instructions', target_lang),
            'Nutritional Values': translation('Nutritional Values (per serving)', target_lang)
        }
        
        translated_recipes = []
        for recipe in recipes:
            translated_recipe = {}
            for key, value in recipe.items():
                translated_key = structure_translations.get(key, key)
                if isinstance(value, list):
                    translated_recipe[translated_key] = [translation(item, target_lang) for item in value if item != '---']
                elif value and value != 'Unnamed Recipe':
                    translated_recipe[translated_key] = translation(value, target_lang)
                else:
                    translated_recipe[translated_key] = value
            translated_recipes.append(translated_recipe)
        
        return translated_recipes, structure_translations, 200
    except Exception as e:
        return None, None, str(e)

def audio_versions(text, lan, recipe_number):
    tts = gTTS(text=text, lang=lan, slow=False)
    audio_path = f'recipe_{recipe_number}.mp3'
    tts.save(audio_path)
    return audio_path

# Make sure to export all necessary functions
__all__ = ['main_model', 'message', 'upload', 'process_image_with_yolo', 'generate_recipe', 'audio_versions', 'translation']