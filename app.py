import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image

# Charger le modèle
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="Classification Poubelles", layout="wide")

# --- STYLE : fond blanc, titre vert, texte noir ---
st.markdown("""
    <style>
        .stApp {
            background-color: white !important;
        }

        /* Titre principal */
        h1 {
            color: #15a000 !important; /* vert */
            font-weight: 900 !important;
        }

        /* Tous les autres textes */
        h2, h3, h4, h5, h6, p, label, span, div {
            color: black !important;
        }

        /* Couleur du texte dans les infos Streamlit */
        .stAlert, .stInfo {
            color: black !important;
        }
        
         /* Cibler le bloc principal du file_uploader */
    div[data-testid="stFileUploader"] {
        background-color: #15a000 !important;     /* vert */
        border-radius: 12px;
        padding: 20px;
    }

    /* Texte à l’intérieur (drag & drop) */
    div[data-testid="stFileUploader"] * {
        color: white !important;                  /* blanc */
        font-weight: 600;
    }

    /* Bouton "Browse files" */
    div[data-testid="stFileUploader"] button {
        background-color: white !important;
        color: #15a000 !important;
        font-weight: 700;
        border-radius: 8px;
    }

    </style>
""", unsafe_allow_html=True)

st.title("🗑️ Classification des poubelles avec YOLO")

# --- Layout en colonnes ---
col_upload, col_original, col_yolo = st.columns([1, 2, 2])

with col_upload:
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

TEMP_DIR = "temp_streamlit"
os.makedirs(TEMP_DIR, exist_ok=True)

def clear_temp():
    for f in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, f))

if uploaded_file:
    clear_temp()

    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with col_original:
        st.subheader("Image originale")
        st.image(Image.open(input_path), use_container_width=True)

    results = model.predict(input_path, imgsz=320)
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        cls_id = int(result.boxes.cls[0])
        prediction = result.names[cls_id]
    else:
        prediction = "Aucun objet détecté"

    output_path = os.path.join(TEMP_DIR, "yolo_" + uploaded_file.name)
    result.save(filename=output_path)

    with col_yolo:
        st.subheader("🎯 Prédiction YOLO")
        st.image(output_path, use_container_width=True)
        st.markdown(f"### Résultat : **{prediction}**")

else:
    with col_original:
        st.info("Importez une image pour afficher la preview.")
    with col_yolo:
        st.info("L’image annotée YOLO apparaîtra ici.")
