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
    prompt = f"""Create {recipe_count} nutritional, delightful and concise recipes using the following vegetables. Each recipe must follow the exact structure below:

                |Recipe Number|: [Number]
                |Recipe Name|: [Name of the dish]
                |Ingredients|:
                [List of ingredients with quantities]
                
                |Cooking Instructions|:
                1. [First step]
                2. [Second step]
                3. [...]
                
                |Nutritional Values| (per serving):
                - Calories: [value]
                - Protein: [value]g
                - Carbohydrates: [value]g
                - Fat: [value]g
                - Fiber: [value]g
                
                Available Ingredients:
                """
                
    for vegetable, count in vegetable_dict.items():
        prompt += f"- {vegetable} ({count} {'piece' if count == 1 else 'pieces'})\n"
                
    prompt += """
                Note:
                1. Use ONLY the vegetables listed above in your recipes.
                2. Be creative with vegetable combinations while ensuring delicious results.
                3. Provide clear, concise cooking instructions.
                4. Include accurate nutritional information for each recipe.
                5. Ensure each recipe is unique and different from the others.
                6. Strictly adhere to the given structure for each recipe.
                7. Seperate the recipes with dashed lines.
                """

    return prompt


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



def translation(i,target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    translated_text = translator.translate(i, src='en', dest=target_lang)
    return translated_text.text
    
                

def extract_recipes(text):
    # Split the text into individual recipes
    recipes = re.split(r'\|Recipe Number\|:', text)[1:]
    
    processed_recipes = []
    
    for recipe in recipes:
        recipe_dict = {}
        
        # Extract Recipe Number
        recipe_dict['Recipe Number'] = re.search(r'(\d+)', recipe).group(1)
        
        # Extract Recipe Name
        recipe_dict['Recipe Name'] = re.search(r'\|Recipe Name\|:\s*(.+?)\s*\|', recipe).group(1)
        
        # Extract Ingredients
        ingredients = re.search(r'\|Ingredients\|:(.*?)\|Cooking Instructions\|:', recipe, re.DOTALL).group(1)
        recipe_dict['Ingredients'] = [ing.strip() for ing in ingredients.strip().split('\n')]
        
        # Extract Cooking Instructions
        instructions = re.search(r'\|Cooking Instructions\|:(.*?)\|Nutritional Values\|', recipe, re.DOTALL).group(1)
        recipe_dict['Cooking Instructions'] = [inst.strip() for inst in instructions.strip().split('\n')]
        
        # Extract Nutritional Values
        nutritional_values = re.search(r'\|Nutritional Values\|\s*\(per serving\):(.*?)$', recipe, re.DOTALL).group(1)
        recipe_dict['Nutritional Values'] = [nv.strip() for nv in nutritional_values.strip().split('\n')]
        
        processed_recipes.append(recipe_dict)
    
    return processed_recipes
                
def generate_recipe(recipe_count, vegetable_dict, target_lang):
    res = generate_recipe_prompt(recipe_count, vegetable_dict)
    gt = model(res)
    recipes = extract_recipes(gt)
    
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


def audio_versions(text, lan, iter):
    tts = gTTS(text=text, lang=lan)
    audio_path = f'recipe_{iter}.wav'
    tts.save(audio_path)
    return audio_path