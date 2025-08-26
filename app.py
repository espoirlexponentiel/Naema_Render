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
MODEL_SUBDIR = os.path.join(MODEL_DIR, "results")  # Dossier contenant les fichiers du modèle
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

MODEL_DRIVE_ID = "1fp-ChRMyJTgzEPgTgBWEGm1lhbEhO1Qw"
ENCODER_DRIVE_ID = "1bSAgS4-RsaFekdU4Qc9wrdw-pXugdbbq"

def download_files():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_SUBDIR):
        st.info("📥 Téléchargement du modèle depuis Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}", "results.tar", quiet=False)

        # Extraction à la racine
        os.system("tar -xf results.tar")

        # Déplacement des fichiers extraits vers MODEL_SUBDIR
        if os.path.exists("results"):
            os.makedirs(MODEL_SUBDIR, exist_ok=True)
            os.system(f"mv results/* {MODEL_SUBDIR}")
            os.system("rm -r results")  # Nettoyage
        else:
            st.error("❌ Le dossier 'results' est introuvable après extraction.")

    if not os.path.exists(ENCODER_PATH):
        st.info("📥 Téléchargement de l'encodeur...")
        gdown.download(f"https://drive.google.com/uc?id={ENCODER_DRIVE_ID}", ENCODER_PATH, quiet=False)

    # Vérification des fichiers critiques
    # required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
    # missing = [f for f in required_files if not os.path.exists(os.path.join(MODEL_SUBDIR, f))]
    # if missing:
    #     st.warning(f"⚠️ Fichiers manquants dans le modèle : {', '.join(missing)}")

download_files()

# -------------------------------
# 🔄 Charger le modèle CamemBERT
# -------------------------------

tokenizer = CamembertTokenizer.from_pretrained(os.path.join(MODEL_SUBDIR, "results"))
model = CamembertForSequenceClassification.from_pretrained(os.path.join(MODEL_SUBDIR, "results"))
label_encoder = joblib.load(ENCODER_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

   

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

        if 'Activité' not in df.columns:
            st.error("Le fichier doit contenir une colonne nommée 'Activité'.")
        else:
            if st.button("Prédire toutes les activités du fichier"):
                df['Classe_NAEMA'] = df['Activité'].apply(lambda x: predict_class(str(x)))
                st.success("✅ Prédiction terminée !")
                st.dataframe(df.head())

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
