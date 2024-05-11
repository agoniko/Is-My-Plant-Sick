import streamlit as st
from PIL import Image
import sys
import streamlit.components.v1 as components
import numpy as np
import cv2

sys.path.append("resources")
sys.path.append("lib")
from gallery_handler import GalleryHandler

st.set_page_config(
    page_title="Is my plant sick?",
    page_icon="🌿",
    layout="wide",
)
st.header("Scroll through our gallery")
height = 400

handler = GalleryHandler(only_test=True)
options_container = st.container()


col1, col2 = options_container.columns(spec=[0.5, 0.5], gap="large")

with col1:
    # filter elements
    selected_leaf = st.selectbox("Filter by leaf type", handler.get_leaf_types())
    handler.filter(selected_leaf, "All")

with col2:
    selected_disease = st.selectbox(
        "Filter by disease type", handler.get_disease_types()
    )
    handler.filter(selected_leaf, selected_disease)

import base64

images_div = []
paths = handler.get_images_path()
for index, path in enumerate(paths):
    file = open(path, "rb")
    contents = file.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file.close()
    images_div.append(
        '<div class="image-item" onclick="send(\'{}\')"><img src="data:image/gif;base64,{}" width="100%"></div>'.format(
            path, data_url
        )
    )

images_div = "".join(images_div)

html_content = """
<!DOCTYPE html>
<html lang="en">
<script>
  function send(path) {{
    const apiUrl = "http://127.0.0.1:8000/path/";
    const data = {{ "path": path }};
    console.log(JSON.stringify(data));

    return fetch(apiUrl, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(data),
    }});
}}
</script>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Grid with Horizontal Scroll</title>
  <style>
    body {{
      margin: 0px;
      padding: 0px;
    }}

.image-grid {{
  display: grid;
  grid-gap: 5px;
  grid-auto-flow: column;
  grid-auto-columns: 20%;
  grid-template-rows: 150px 150px;
  overflow-x: scroll;
  height: 300; /* Set the overall height for the grid */
  background-color: rgb(15, 17, 22); /* Light gray background color */
}}

.image-item {{
  width: 100%;
  height: 100%;
  background-color: lightgray;
  transition: transform 0.3s ease; /* Add a smooth transition effect */
}}

.image-item:hover {{
  transform: scale(1.1); /* Zoom in a little on hover */
}}

.image-item img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
}}
  </style>
</head>
<body>
<div class="image-grid">
{}
</div>
</body>
</html>
""".format(
    images_div
)


components.html(html_content, height=300)

# read the last line from path.txt
with open("backend/path.txt", "r") as f:
    last_line = f.readlines()[-1]

prev = last_line
try:
    index = np.where(np.array(paths) == last_line.strip())[0][0]
except:
    index = 0
data_container = st.container()
image_col, description_col = data_container.columns(spec=[0.5, 0.5], gap="large")

with image_col:
    (
        image,
        attention,
        symptoms,
        treatment,
        leaf,
        disease,
        correct,
        score,
    ) = handler.get_image(index)

    image = image.resize((int(image.width / image.height * height), height))
    st.image(image, use_column_width="auto")

with description_col:
    with st.expander("Show affected areas (Beta feature)"):
        attention = cv2.resize(attention, (int(image.width / image.height * height), height))
        st.image(attention, use_column_width="auto")
    leaf_color = "green" if correct[0] else "red"
    disease_color = "green" if correct[1] else "red"
    st.write(f"## Leaf type: :{leaf_color}[{leaf.replace('_', ' ')}]")
    st.write(f"## Disease type: :{disease_color}[{disease.replace('_', ' ')}]")
    if disease != "healthy":
        st.write(f"**Symptoms:** {symptoms}")
        st.write(f"**Treatment:** {treatment}")
    else:
        st.write(f"")
        st.write(f"")


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

while prev == last_line:
    with open("backend/path.txt", "r") as f:
        last_line = f.readlines()[-1]


st.rerun()
