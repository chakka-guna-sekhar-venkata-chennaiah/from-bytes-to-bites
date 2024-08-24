import numpy as np
import pandas as pd
from gtts import gTTS
from ultralytics import YOLO
import streamlit as st
import cv2
import time
import base64
import time
import shutil
import os
from PIL import Image
import base64
import random
from utils import main_model,message,upload,process_image_with_yolo,generate_recipe,audio_versions

heading_styles = '''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bungee+Shade&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Indie+Flower&display=swap');

        .glowing-heading {
            font-family: 'Poppins', sans-serif;
            font-size: 48px;
            text-align: center;
            animation: glowing 2s infinite;
            color: #FF5733; /* Orange color */
            text-shadow: 2px 2px 4px #333;
        }

        .sub-heading {
            font-family: 'Quicksand', cursive;
            font-size: 32px;
            text-align: center;
            animation: colorChange 4s infinite;
            text-shadow: 1px 1px 2px #333;
            color: #0099CC; /* Blue color */
        }

        @keyframes glowing {
            0% { color: #FF5733; } /* Orange color */
            25% { color: #FFFFFF; } /* White color */
            50% { color: #128807; } /* Green color */
            75% { color: #0000FF; } /* Blue color */
            100% { color: #FF5733; } /* Orange color */
        }

        @keyframes colorChange {
            0% { color: #0099CC; } /* Blue color */
            25% { color: #FF5733; } /* Orange color */
            50% { color: #66FF66; } /* Light Green color */
            75% { color: #FFCC00; } /* Yellow color */
            100% { color: #0099CC; } /* Blue color */
        }
    </style>
'''

# Set page config
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon='🤖', page_title='From Bytes to Bites')

# Display the custom heading styles
st.markdown(heading_styles, unsafe_allow_html=True)

# Create the headings
st.markdown(f'<p class="glowing-heading">🤖 From Bytes To Bites 🤖</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-heading">Deep Learning and AI-Generated Nutritional Recipes for Superhero Moms (Multilingual Support)</p>', unsafe_allow_html=True)

# Image
st.image('working.jpg',use_column_width=True)



       

 
#sidebar_option = st.sidebar.radio("Select an option", ("Take picture for prediction"))

def main():
    
    
    
    
   
    
    if st.checkbox('Take a picture for Recipe & Audio Generation'):
    
        
        image, original_image,image_filename= upload()
        if original_image is not None and image_filename is not None and len(image_filename)!=0 and st.checkbox('Start Identifying Veggies!!'):  # Check if original_image is not None
            st.info('Wait for the results...!')
                #image1=cv2.imread(image)
            pic0=image
            uniquelist=process_image_with_yolo(pic0)
            if uniquelist:
                vegetables=uniquelist.keys()
                counts=uniquelist.values()
                data={
                    'Veggie':vegetables,
                    'Counts':counts
                }   
                df=pd.DataFrame(data)
                st.write(df)
                lan_dcit={
                        'Telugu':'te',
                        'Malayalam':'ml',
                        'Hindi':'hi',
                        'Kannada':'kn',
                        'Tamil':'ta',
                        'English':'en',
                        'Gujarati':'gu',
                        'Punjabi':'pa',
                        'Bengali':'bn'
                    }
                recip_dict={
                    '1':'one',
                    '2':'two',
                    '3':'three'
                }
                choices=['Telugu','Malayalam','Hindi','Kannada','Tamil','English','Gujarati','Punjabi','Bengali']
                language=st.selectbox('Choose the language in which you want the recipe?',choices)
                recipe=st.selectbox('How many different types of recipes you want??',['1','2','3'])
                frecipe=recip_dict[recipe]
                if st.button('Generate Recipes & Audio'):
                    
                    final_result=generate_recipe(recipe,uniquelist,lan_dcit[language])
                    for i in range(len(final_result)):
                        st.write(final_result[i])
                    """
                    #for i in range(recip_dict[recipe],recip_dict[recipe]+1):
                        #for i in recipe_paragraphs:
                            #st.write(i)
                        #st.write('-'*100)
                    text_to_speech = final_result
                    tts = gTTS(text=text_to_speech, lang=lan_dcit[language])
                    
                        # Save the audio file
                    audio_path = 'saved_audio.wav'
                    tts.save(audio_path)
                    
                        # Play the audio
                    st.balloons()
                    with st.spinner('Wait for the audio version................'):
                        time.sleep(3)
                    st.info('Aduio Version of the Recipes')
                    st.audio(audio_path, format='audio/wav')
                   
                    """  

                    
                      
                                                
            else:
                message()
                
        
            

   
if __name__ == '__main__':
    
   
    main()
