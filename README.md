<div align='center'>
<h1>
🤖 From Bytes to Bites 🤖
</h1>
<h2>
Deep Learning & AI Generated Recipes
</h2>
</div>

## 1. 🖊️ __Description__
`From Bytes to Bites` is an software web application that generates recipes and its audio version is user interested language.
## 2. 🏗️ __Architecture__
![image](https://github.com/user-attachments/assets/f384925e-6e24-4df3-bf78-47de01c01318)
## 3. ⚙️ __Local Setup__
1. Setup an API KEY on [Google's Maker Suite]([text](https://aistudio.google.com/))
2. Clone the repo
   ```
   git clone https://github.com/chakka-guna-sekhar-venkata-chennaiah/from-bytes-to-bites.git
   ```
3. Navigate to the cloned path and create a folder name `.streamlit` in the project root directory. The directory strucutre as follows:-
    ```
    from-bytes-to-bites-v1
        -- .streamlit
        -- README.md
        -- app.py
        -- best.pt
        -- utils.py
        -- working.jpg
    ```
4. Create a new file named `secrets.toml` under `.streamlit` folder as follows:-
   ```
   #secrets.toml file should be followed as
   api_key='XXXXXXXX'

   #Directory structure as follows

   from-bytes-to-bites-v1
    -- .streamlit
            -- secrets.toml
    -- README.md
    -- app.py
    -- best.pt
    -- utils.py
    -- working.jpg
    ```
5. Installing the dependencies
   ```
   cd from-bytes-to-bites-v1/
   python -m pip install -r requirements.txt

   #or

   pip install -r requirements.txt
   ```
6. It's time to start the application with following command
   ```
   python -m streamlit run app.py

   #or

   streamlit run app.py
   ```
    😄 Enjoy the application 💗
<br>

+ [Official Cloud Application](https://from-bytes-to-bites-v1.streamlit.app/)
  

