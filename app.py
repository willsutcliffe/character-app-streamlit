import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import os
from PIL import Image
import torch
from torchvision import datasets, transforms, models 
import pickle
from models import Net
import numpy as np
import random




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
        model = Net(75)
        path = "./assets/hirigana_pytorch.pth" 
        with open('assets/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
    else:
        model = Net(46)
        path = "./assets/katakana_pytorch.pth"
        with open('assets/katakana_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
else:
    option_rus = st.selectbox("Select upper or lower case", ("Upper", "Lower"))
    size = 64

    if option_rus == "Upper case letter":
        model = Net(31)
        path = "./assets/CYRILLIC_pytorch_aug.pth"
        labels = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ы', 'Э', 'Ю', 'Я']
    else:
        model = Net(33)
        path = "./assets/cyrillic_pytorch_aug.pth"
        labels = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']


model.load_state_dict(torch.load(path, map_location="cpu"))
model.eval()


data_aug_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
])


col1, col2 = st.columns([3,2],gap="small")
with col1:
    sentence = st.text(f'Draw the character {random.choice(labels)}')
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
with col2:
    sentence = st.text('Do you want to see an example?')
    img = Image.open('0_0_1378_20190729081444.png')
    img = np.array(img)
    img = Image.fromarray(255 - img)
    st.image(img, caption=None, width=100)

def get_prediction(image):
    if option_lang == "Russian":
        image = Image.fromarray(255 - image[:,:,:3])
        image = data_aug_transform(image)
    else:
        image = Image.fromarray(image)
        image = image.resize((size,size))
        image = np.array(image)[:,:,:3]/255.
        image = np.mean(image,axis=2)
        image = torch.tensor(image,dtype=torch.float)
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
