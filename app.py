import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        revision="fp16",
        use_safetensors=True
    ).to("cuda")
    return pipe

pipe = load_model()

# Generate images
def generate_images(prompt, count):
    images = []
    for _ in range(count):
        image = pipe(prompt).images[0]
        images.append(image)
    return images

# Make a grid
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

# UI
st.title("🧠 AI Fashion Moodboard Generator 👕")
prompt = st.text_input("Enter fashion theme (e.g. 'tropical jungle kidswear'):", "pastel jungle kidswear")
num = st.slider("Number of Images", 4, 9, step=1, value=9)

if st.button("Generate Moodboard"):
    with st.spinner("Generating..."):
        images = generate_images(prompt, num)
        rows = cols = int(num ** 0.5)
        grid_image = make_grid(images, rows, cols)
        st.image(grid_image, caption="Generated Moodboard", use_column_width=True)
