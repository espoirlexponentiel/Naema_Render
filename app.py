import streamlit as st
import joblib
import os
import gdown
import torch
import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from io import BytesIO

# -------------------------------
# 🔐 Sécurité : mot de passe
# -------------------------------
PASSWORD = os.environ.get("PASSWORD", "naema2025")  # à définir dans Render
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("Mot de passe :", type="password")
    if pwd == PASSWORD:
        st.session_state.authenticated = True
        st.success("✅ Authentification réussie !")
    else:
        st.stop()

# -------------------------------
# 📂 Téléchargement du modèle
# -------------------------------
MODEL_DIR = "model_naema"
MODEL_PATH = os.path.join(MODEL_DIR, "results/model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ⚠️ Remplace par tes vrais IDs Google Drive
MODEL_DRIVE_ID = "https://drive.google.com/drive/folders/1qsWhhkeYCGyx-fGMUV_Ew7pGRW0Om3J9?usp=drive_link"      # ex: results compressé en .tar.gz ou .zip
ENCODER_DRIVE_ID = "https://drive.google.com/file/d/1bSAgS4-RsaFekdU4Qc9wrdw-pXugdbbq/view?usp=drive_link"   # ex: label_encoder.pkl

def download_files():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.info("Téléchargement du modèle depuis Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}", "model.tar.gz", quiet=False)
        os.system(f"tar -xzf model.tar.gz -C {MODEL_DIR}")

    if not os.path.exists(ENCODER_PATH):
        st.info("Téléchargement de l'encodeur...")
        gdown.download(f"https://drive.google.com/uc?id={ENCODER_DRIVE_ID}", ENCODER_PATH, quiet=False)

download_files()

# -------------------------------
# 🔄 Charger le modèle CamemBERT
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_PATH)
    model = CamembertForSequenceClassification.from_pretrained(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, label_encoder, device

tokenizer, model, label_encoder, device = load_model()

# -------------------------------
# 🔹 Fonction de prédiction
# -------------------------------
def predict_class(intitule):
    inputs = tokenizer(
        intitule,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return label_encoder.inverse_transform([predicted_class_id])[0]

# -------------------------------
# 🖥️ Interface Streamlit
# -------------------------------
st.title("📊 Classification NAEMA")
st.write("Entrez une activité économique ou téléchargez un fichier Excel pour obtenir les classes NAEMA.")

# ---- Prédiction simple texte ----
activite = st.text_area("Activité :", placeholder="Ex: Culture de maïs, Commerce de détail...")

if st.button("Prédire l'activité unique"):
    if activite.strip() == "":
        st.warning("⚠️ Merci de saisir une activité.")
    else:
        try:
            classe = predict_class(activite)
            st.success(f"✅ Classe NAEMA prédite : **{classe}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ---- Prédiction sur fichier Excel ----
st.subheader("📁 Prédire depuis un fichier Excel")
uploaded_file = st.file_uploader("Télécharger un fichier Excel (.xlsx ou .xls)", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Aperçu du fichier :")
        st.dataframe(df.head())

        # Vérifier qu'il y a une colonne 'Activité'
        if 'Activité' not in df.columns:
            st.error("Le fichier doit contenir une colonne nommée 'Activité'.")
        else:
            if st.button("Prédire toutes les activités du fichier"):
                df['Classe_NAEMA'] = df['Activité'].apply(lambda x: predict_class(str(x)))
                st.success("✅ Prédiction terminée !")
                st.dataframe(df.head())

                # Préparer le téléchargement
                output = BytesIO()
                df.to_excel(output, index=False)
                output.seek(0)
                st.download_button(
                    label="📥 Télécharger le fichier prédit",
                    data=output,
                    file_name="resultats_naema.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")