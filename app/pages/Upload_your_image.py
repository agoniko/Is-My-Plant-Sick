import streamlit as st
from PIL import Image
import pandas as pd
import sys
import cv2

sys.path.append("lib")
from ai_engine import AIEngine

st.set_page_config(
    page_title="Is my plant sick?",
    page_icon="🌿",
    layout="wide",
)
st.header("Upload your Images")

model = AIEngine()
classes = pd.read_csv("resources/classes.csv")
classes["leaf"] = classes["class"].apply(lambda x: x.split("___")[0])
classes["disease"] = (
    classes["class"].apply(lambda x: x.split("___")[1]).str.replace("_", " ")
)
descriptions = pd.read_csv("resources/disease_description.csv")

col1, col2 = st.columns(spec=[0.5, 0.5], gap="large")
with col1:
    st.write(
        "### Our model can recognize plants and diseases as shown in the table below"
    )
    st.write(classes[["leaf", "disease"]].groupby("leaf").agg(list))

with col2:
    uploaded = st.file_uploader(
        "Upload your image here",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

width = 400
if len(uploaded) > 0:
    for image in uploaded:
        image = Image.open(image)
        image, label, score, attention = model.predict(image)

        leaf = classes.iloc[label]["leaf"]
        disease = classes.iloc[label]["disease"]

        col1, col2 = st.columns(spec=[0.5, 0.5], gap="large")
        with col1:
            st.write(
                f"#### Leaf type: {leaf.replace('_', ' ')}, Disease: {disease.replace('_', ' ')}\n Confidence: {score * 100:.2f}%"
            )
            st.image(image, use_column_width=True)
        with col2:
            with st.expander("Show affected areas (Beta feature)"):
                st.image(attention, use_column_width=True)
            if disease not in ["healthy", "background"]:
                st.write("#### Symptoms:")
                st.write(descriptions[descriptions["disease"] == disease.replace(" ", "_")]["description"].values[0])
                st.write("#### Treatment:")
                st.write(descriptions[descriptions["disease"] == disease.replace(" ", "_")]["treatment"].values[0])



end_page = """---
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
© 2024 Agoniko. All rights reserved.

---"""
st.markdown(end_page, unsafe_allow_html=True)