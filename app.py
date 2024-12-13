
import streamlit as st
from PIL import Image
image = Image.open("images.jpg")
st.image(image, caption="Iris Flower", width=400)
st.write(f"## Iris Classifier")
x1 = st.text_input("Sepal Length")
x2 = st.text_input("Sepal Width")
x3 = st.text_input("Petal Length")
x4 = st.text_input("Petal Width")


import pickle
import numpy as np
model = pickle.load(open("DT_model.sav", "rb"))

if st.button("Predict"):
    input_data = np.array([x1,x2,x3,x4], dtype=float).reshape(1,-1)
    result = model.predict(input_data)
    if result[0]==0:
        output_class = "Iris setosa"
    elif result[0]==1:
        output_class = "Iris virginica"
    else:
        output_class = "Iris versicolor"
    st.write(f"The class of the flower is: {output_class}")