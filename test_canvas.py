
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.title("Canvas Compatibility Test")

canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas",
)

if canvas.image_data is not None:
    st.write("Canvas is working!")
