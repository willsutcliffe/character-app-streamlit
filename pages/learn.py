import numpy as np
from PIL import Image
import streamlit as st
import random
import pickle


st.set_page_config(
page_title = "Learn characters",
page_icon = ":pencil:",
)

if 'total' not in st.session_state:
    st.session_state[f'total'] = 0
    st.session_state[f'correct'] = 0

def generate_choices(labels):
    st.session_state.learnchoice = random.choice(labels)
    col = random.choice([0,1,2,3])
    choices = [random.choice(labels) for i in range(0,4)]
    choices[col] =  st.session_state.learnchoice 
    preds = []
    for i in range(0,4):
        choice = choices[i]
        st.session_state[f'choice{i}'] = choices[i]
    st.session_state[f'total'] +=1
        
option_lang = st.selectbox("Select language ", ("Russian", "Japanese"))
if option_lang == 'Japanese':
    option_jap = st.selectbox("Select script", ("Hiragana", "Katakana"))
    size = 64
    if option_jap == "Hiragana":
        examples_path = "./examples/hiragana/"
        with open('assets/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
    else:
        examples_path = "./examples/katakana/"
        with open('assets/katakana_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
    if 'learnchoice' not in st.session_state:
        generate_choices(labels)
else:
    option_rus = st.selectbox("Select upper or lower case", ("Upper", "Lower"))

    if option_rus == "Upper":
        examples_path = "./examples/cyrillic/large/"
        labels = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ы', 'Э', 'Ю', 'Я']
    else:
        examples_path = "./examples/cyrillic/small/"
        labels = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
    if 'learnchoice' not in st.session_state:
        generate_choices(labels)




if option_lang == "Japanese":
    labels = list(labels.values())
generate = st.button("Next question:", key='next') 
if generate:
    generate_choices(labels)
st.text(f"Which of these is: {st.session_state.learnchoice}:")
col1, col2, col3, col4 = st.columns([2,2,2,2],gap="small")
cols = [col1, col2, col3, col4]
try:            
    for i in range(0,4):
        with cols[i]:
            choice = st.session_state[f'choice{i}']
            img = Image.open(f'{examples_path}/{choice}/0.png')
            img = np.array(img)
            if option_lang == "Japanese":
                img = Image.fromarray(img)
            else:
                img = Image.fromarray(255 - img)
            st.image(img, caption=None, width=100)
            predict = st.button(f"{i}")
            if predict:
                if st.session_state.learnchoice  ==  st.session_state[f'choice{i}']:
                    st.text("Correct")    
                    st.session_state[f'correct'] +=1
                else:
                    st.text("False")    

except:
    st.text(f"Please press 'Next question' above.")

stats = st.button("See Statistics:",key='stats')
if stats:
    st.text(f"{st.session_state[f'correct']}/{ st.session_state[f'total']} correct")




