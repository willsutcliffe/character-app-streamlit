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
        examples_path = "./examples/hiragana/"
        with open('assets/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
    else:
        model = Net(46)
        path = "./assets/katakana_pytorch.pth"
        examples_path = "./examples/katakana/"
        with open('assets/katakana_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
else:
    option_rus = st.selectbox("Select upper or lower case", ("Upper", "Lower"))
    option_aug = st.selectbox("Select model with or without data augmentation:", ("with", "without"))
    size = 64

    if option_rus == "Upper":
        if option_aug == "with":
            model = Net(31)
            path = "./assets/CYRILLIC_pytorch_aug.pth"
            examples_path = "./examples/cyrillic/large/"
            labels = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ы', 'Э', 'Ю', 'Я']
        else:
            model = Net(33)
            path = "./assets/CYRILLIC_pytorch.pth"
            examples_path = "./examples/cyrillic/large/"
            labels = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
          'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
          'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
    else:
        if option_aug == "with":
            model = Net(33)
            path = "./assets/cyrillic_pytorch_aug.pth"
            examples_path = "./examples/cyrillic/small/"
            labels = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
        else:
            model = Net(33)
            path = "./assets/cyrillic_pytorch.pth"
            examples_path = "./examples/cyrillic/small/"
            labels = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
          'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
          'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

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
    examples_option = st.selectbox("Show examples:", ("On", "Off"))
    generate = st.button("Generate task:") 
    if generate:
        if option_lang == "Russian":
            st.session_state.choice = random.choice(labels)
        else:
            st.session_state.choice = random.choice(list(labels.values()))

    try:
        if 'choice' in st.session_state:
            choice = st.session_state.choice
            sentence = st.text(f'Draw the character {choice}')
            if examples_option == "On":
                img = Image.open(f'{examples_path}/{choice}/0.png')
                img = np.array(img)

                if option_lang == "Russian":
                    img = Image.fromarray(255 - img)
                else:
                    img = Image.fromarray(img)
                st.image(img, caption=None, width=120)
    except:
         st.write("Please generate a new exercise.")

def get_prediction(image):
    if option_lang == "Russian":
        image = Image.fromarray(255 - image[:,:,:3])
        image = data_aug_transform(image)
    else:
        image = Image.fromarray(image)
        image = image.resize((size,size))
        if option_lang == "Russian":
            image = 1-np.array(image)[:,:,:3]/255.
        else:
            image = np.array(image)[:,:,:3]/255.
        image = np.mean(image,axis=2)
        image = torch.tensor(image,dtype=torch.float)
        image = torch.unsqueeze(image, dim=0)

    image = torch.unsqueeze(image, dim=0)
    outputs =  model(image)
    pred = torch.argmax(outputs, dim=1) 
    label = labels[int(pred)]
    indices  =  torch.topk(outputs.flatten(), 3).indices
    st.text("Top prediction : {}".format(label))    
    st.text("Was my prediction correct?")    

    st.text(f"If not the next top 2 predictions : {labels[int(indices[1])]}, {labels[int(indices[2])]}")    
    return label, image


if canvas_result.image_data is not None and predict:
    label, img = get_prediction(canvas_result.image_data)
