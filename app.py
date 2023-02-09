import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import os
from PIL import Image
import torch
import pickle
from models import CyrNet, HNet, KNet




labels = ""
path = ""
device='cpu'
size = 32



st.set_page_config(
page_title = "Character Classifier App",
page_icon = ":pencil:",
)

hide_streamlit_style = """
                       <style>
                       #MainMenu {visibility: hidden;}
                       footer {visibility: hidden;}
                       </style>
                       """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Character Classifier App")

option_lang = st.selectbox("Select language ", ("Russian", "Japanese"))
if option_lang == 'Japanese':
    option_jap = st.selectbox("Select script", ("Hiragana", "Katakana"))
    size = 64
    if option_jap == "Hiragana":
        model = HNet()
        path = "./assets/hirigana_pytorch.pth" 
        with open('assets/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
    else:
        model = KNet()
        path = "./assets/katakana_pytorch.pth"
        with open('assets/katakana_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
else:
    model = CyrNet()
    option_rus = st.selectbox("Select upper or lower case", ("Upper", "Lower"))
    labels = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
          'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
          'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
    size = 32

    if option_rus == "Upper case letter":
        path = "./assets/CYRILLIC_pytorch.pth"
    else:
        path = "./assets/cyrillic_pytorch.pth"


model.to(device)
model.load_state_dict(torch.load(path))
model.eval()



stroke_width = st.sidebar.slider("Stroke width: ", 1, 40, 10)

canvas_result = st_canvas(
stroke_width = stroke_width,
stroke_color = "#fff",
background_color = "#000",
height = 280,
width = 280,
drawing_mode = "freedraw",
key = "canvas",
)
predict = st.button("Predict")

def get_prediction(image):
    # get the prediction from your model and return itif canvas_result.image_data is not None and predict:
    st.text("Prediction : {}".format(image.shape))    

    image = Image.fromarray(image)
    image = image.resize((size,size))
    
    image = np.array(image)[:,:,:3]/255.
    image = np.mean(image,axis=2)
#    st.text("Prediction : {}".format(image.shape))    
    image = torch.tensor(image,dtype=torch.float)
    if option_lang == "Japanese":
        image = torch.unsqueeze(image, dim=0)
    image = torch.unsqueeze(image, dim=0)
    outputs =  model(image)
    pred = torch.argmax(outputs, dim=1) 
    label = labels[int(pred)]
    return label, image


if canvas_result.image_data is not None and predict:
    label, img = get_prediction(canvas_result.image_data)
    st.text("Prediction : {}".format(label))    
    st.text("Was my prediction correct?")    
