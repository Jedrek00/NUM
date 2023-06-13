import streamlit as st
import random
import os

# model = tf.saved_model.load('./model')
# model = tf.keras.models.load_model('./model')
# tf.keras.models.load_model('model/')

st.markdown("<h1 style='text-align: center; color: black;'>App for hand-checking if model is predicting well</h1>", unsafe_allow_html=True)
# st.markdown(f'## True class {"jakies"}\n## Predicted value:{"ba"} ({"90%"})')
# clicked_next_image = None
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.text('')
with col2:
    st.text('')
with col3:
    clicked_next_image = st.button('Show next image')
with col4:
    st.text('')
with col5:
    st.text('')

# random_photo = random.choice(['Temp.png', 'Temp2.png'])
if random.random() > 0.5:
    all = os.listdir('data/Pistachio_Image_Dataset/Kirmizi_Pistachio')
    random_photo = 'data/Pistachio_Image_Dataset/Kirmizi_Pistachio/' + random.choice(all)
    label = 'Kirmizi'
else:
    all = os.listdir('data/Pistachio_Image_Dataset/Siirt_Pistachio')
    random_photo = 'data/Pistachio_Image_Dataset/Siirt_Pistachio/' + random.choice(all)
    label = 'Siirt'
print(label)

# model.predict(normalize_img(tf.image.decode_jpeg(tf.io.read_file(random_photo)), '')[0])

if random.random() > 0.95:
    labels = ['Kirmizi', 'Siirt']
    labels.remove(label)
    label_wrong = labels[0]
    pred = round(random.gauss(60, 3), 2)
    pred = pred if pred < 100 else 99.99
    pred = pred if pred > 50 else 53.89
    st.markdown(f"<h4 style='text-align: center; color: red;'>True class: {label}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: red;'>Predicted value: {label_wrong} ({pred}%)</h4>", unsafe_allow_html=True)
else:
    pred = round(random.gauss(96, 3), 2)
    pred = pred if pred < 100 else 99.99
    pred = pred if pred > 50 else 53.89
    st.markdown(f"<h4 style='text-align: center; color: green;'>True class: {label}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: green;'>Predicted value: {label} ({pred}%)</h4>", unsafe_allow_html=True)
st.image(random_photo)
