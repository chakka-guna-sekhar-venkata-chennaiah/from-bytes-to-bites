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
    prompt = f"""Create {recipe_count} nutritional, delightful and concise recipes using ONLY the following vegetables. Each recipe MUST strictly follow this exact structure:

    Recipe Number: [number]
    Recipe Name: [name]
    Ingredients:
    - [ingredient 1 with quantity]
    - [ingredient 2 with quantity]
    ...
    Cooking Instructions:
    1. [step 1]
    2. [step 2]
    ...
    Nutritional Values (per serving):
    - Calories: [value]
    - Protein: [value]g
    - Carbohydrates: [value]g
    - Fat: [value]g
    - Fiber: [value]g

    Available vegetables:
    """
    
    for vegetable, count in vegetable_dict.items():
        prompt += f"- {vegetable} ({count} {'piece' if count == 1 else 'pieces'})\n"
    
    prompt += """
    Notes:
    1. Use ONLY the vegetables listed above in your recipes.
    2. Be creative with vegetable combinations while ensuring delicious results.
    3. Provide clear, concise cooking instructions.
    4. Include accurate nutritional information for each recipe.
    5. Ensure each recipe is unique and different from the others.
    6. Strictly adhere to the given structure for each recipe.
    7. Separate the recipes with a line of dashes (---).
    """
    
    return prompt

def parse_recipes(text):
    recipes = []
    raw_recipes = re.split(r'\n-{3,}\n', text)
    
    for raw_recipe in raw_recipes:
        recipe = {
            'Recipe Number': '',
            'Recipe Name': '',
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
            elif line == 'Nutritional Values (per serving):':
                current_section = 'Nutritional Values'
            elif current_section:
                if current_section == 'Cooking Instructions':
                    # Remove numbering from instructions
                    line = re.sub(r'^\d+\.\s*', '', line)
                recipe[current_section].append(line)
        
        if recipe['Recipe Name']:  # Only add non-empty recipes
            recipes.append(recipe)
    
    return recipes

def model(recipe_prompt):
    client = OpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key= st.secrets['key']
    )
    response = client.chat.completions.create(
                                                model="llama-3.1-70b-versatile",
                                                messages=[
                                                    {"role": "user", "content":recipe_prompt},
                                                ]
                                                )

    return response.choices[0].message.content
def generate_recipe(recipe_count, vegetable_dict, target_lang):
    prompt = generate_recipe_prompt(recipe_count, vegetable_dict)
    response = model(prompt)  # Assuming you have a model function that generates the recipes
    recipes = parse_recipes(response)
    
    translated_recipes = []
    for recipe in recipes:
        translated_recipe = {}
        for key, value in recipe.items():
            if isinstance(value, list):
                translated_recipe[key] = [translation(item, target_lang) for item in value]
            else:
                translated_recipe[key] = translation(value, target_lang)
        translated_recipes.append(translated_recipe)
    
    return translated_recipes

def translation(i,target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    translated_text = translator.translate(i, src='en', dest=target_lang)
    return translated_text.text
    
                

                
def audio_versions(text, lan, iter):
    tts = gTTS(text=text, lang=lan)
    audio_path = f'recipe_{iter}.wav'
    tts.save(audio_path)
    return audio_path