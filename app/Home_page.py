import streamlit as st
import os

description_path = "resources/home_description.md"

def read_markdown_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return None




st.set_page_config(
    page_title="Is my plant sick?",
    page_icon="🌿",
)

markdown_content = read_markdown_file(description_path)
if markdown_content is not None:
    st.markdown(markdown_content, unsafe_allow_html=True)





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