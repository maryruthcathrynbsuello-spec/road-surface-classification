from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageOps

st.set_page_config(
    layout="wide",
)

PHOTO_SIZE = 180         
CLASS_CARD_HEIGHT = 170  


def team_photo(path, size=PHOTO_SIZE):
    if Path(path).exists():
        try:
            img = Image.open(path).convert("RGB")
            return ImageOps.fit(img, (size, size), centering=(0.5, 0.4))
        except OSError:
            pass

def priority_table(rules):
    with st.container(border=True):
        head = st.columns([1, 3])
        head[0].markdown("**Priority**")
        head[1].markdown("**Road Condition**")
        for rule in rules:
            st.markdown(
                '<hr style="margin: 0.25rem 0; opacity: 0.25;">',
                unsafe_allow_html=True,
            )
            row = st.columns([1, 3], vertical_alignment="center")
            with row[0]:
                st.badge(rule["priority"], icon=":material/circle:", color=rule["color"])
            row[1].write(rule["road_condition"])

OBJECTIVES = [
    "Classify road-surface images into six categories: crack, pothole, damaged asphalt, water-filled pothole, open manhole, and normal road surface.",
    "Build a model that automatically identifies road-surface conditions from images using a Convolutional Neural Network (CNN).",
    "Evaluate the model using unseen test images to measure its classification performance.",
]

CLASSES = [
    ("Crack", "Road surfaces showing visible cracks or fractures."),
    ("Pothole", "Depressions or holes formed in the road surface."),
    ("Damaged Asphalt", "Road surfaces with deterioration or damage to the asphalt."),
    ("Water-Filled Pothole", "Potholes containing standing water."),
    ("Open Manhole", "Road areas with an exposed/open manhole."),
    ("Normal Road Surface", "Road surfaces without the target damage conditions."),
]

PRIORITY_RULES = [
    {"priority": "Urgent", "color": "red", "road_condition": "Open Manhole"},
    {"priority": "High", "color": "orange", "road_condition": "Pothole, Water-Filled Pothole"},
    {"priority": "Medium", "color": "yellow", "road_condition": "Damaged Asphalt, Crack"},
    {"priority": "None", "color": "green", "road_condition": "Normal Road Surface"},
]

LIMITATIONS = [
    "The Normal Road Surface class was generated from unlabeled road areas rather than being directly labeled as a normal class.",
    "The Open Manhole class has very few examples (approximately 11 test images), which can make its evaluation metrics less reliable.",
    "The model's performance depends on the quality and distribution of the dataset.",
    "The application should flag predictions with low confidence for human review rather than treating them as certain classifications.",
    "The maintenance priority system is a team-defined assumption and does not represent an official road-maintenance standard.",
    "The model should be evaluated on unseen test images to assess how well it generalizes beyond the training data.",
]

DATASET_CITATION = "Moni, K., Ahmed, T., Das, N., Ripa, M. T. A., Saha, S. K., Supti, S. J., & Noor, J. (2026). Multi-class Road-Surface Condition Dataset for Automated Pothole and Road-Damage Classification. Mendeley Data. https://data.mendeley.com/datasets/c43y93xswd/1"

TEAM = [
    {"name": "Mary Ruth Cathryn Suello", "role": "Backend Developer", "photo": "app/assets/suello.jpg"},
    {"name": "Jhiro Bautista", "role": "Frontend Developer", "photo": "app/assets/bbm.jpg"},
    {"name": "Elicxia Geighñel Mistica", "role": "Frontend Developer", "photo": "app/assets/mistica.png"},
]

st.title("About")
st.write("RoadWatch is a road-surface classification system designed to automatically identify different road conditions " \
        "from images using a Convolutional Neural Network (CNN). The system classifies road surfaces into six categories: crack, " \
        "pothole, damaged asphalt, water-filled pothole, open manhole, and normal road surface. It aims to support road condition " \
        "monitoring by providing a faster and more consistent way of identifying potential road damage and determining its maintenance " \
        "priority. The system also evaluates the model using unseen test images and presents classification results through a Streamlit-" \
        "based application.")

st.header("Project Objectives")
for item in OBJECTIVES:
    st.markdown(f"- {item}")

st.divider()

st.header("Road Surface Classes")
for row_start in range(0, len(CLASSES), 3):
    cols = st.columns(3)
    for col, (name, desc) in zip(cols, CLASSES[row_start : row_start + 3]):
        with col.container(border=True, height=CLASS_CARD_HEIGHT):
            st.subheader(name)
            st.write(desc)

st.divider()

st.header("Maintenance Priority Rules")
st.write("How each predicted class maps to a maintenance priority.")
priority_table(PRIORITY_RULES)

st.divider()

st.header("Limitations")
for item in LIMITATIONS:
    st.markdown(f"- {item}")

st.divider()

# ----- Dataset citation -----
st.header("Dataset Citation")
st.markdown(f"> {DATASET_CITATION}")

st.divider()

st.header("Our Team")
per_row = 4  
for row_start in range(0, len(TEAM), per_row):
    cols = st.columns(per_row, gap="large")
    for col, member in zip(cols, TEAM[row_start : row_start + per_row]):
        with col:
            st.image(team_photo(member["photo"]), width=PHOTO_SIZE)
            st.markdown(f"**{member['name']}**")
            st.caption(member["role"])