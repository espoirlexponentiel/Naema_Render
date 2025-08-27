import streamlit as st
import joblib
import os
import gdown
import torch
import pandas as pd
import tarfile
import tempfile
import shutil
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from io import BytesIO

# st.text(f"Transformers version : {transformers.__version__}")

# # -------------------------------
# # 🔐 Sécurité : mot de passe
# # -------------------------------
# PASSWORD = os.environ.get("PASSWORD", "naema2025")
# if "authenticated" not in st.session_state:
#     st.session_state.authenticated = False

# if not st.session_state.authenticated:
#     pwd = st.text_input("Mot de passe :", type="password")
#     if pwd == PASSWORD:
#         st.session_state.authenticated = True
#         st.success("✅ Authentification réussie !")
#     else:
#         st.stop()

# -------------------------------
# 📂 Paramètres de téléchargement
# -------------------------------
MODEL_DIR = "C:\model_naema"
MODEL_SUBDIR = os.path.join(MODEL_DIR, "results/results")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

MODEL_DRIVE_ID = "1vpRfWVAzgsyAlIWyobWAGVylNWG__qpF"
ENCODER_DRIVE_ID = "1bSAgS4-RsaFekdU4Qc9wrdw-pXugdbbq"

# -------------------------------
# 📥 Téléchargement des fichiers
# -------------------------------
def download_tar_from_drive(file_id, output_path="results.tar"):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return output_path

def extract_tar_to_temp(tar_path):
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=temp_dir)
    return temp_dir

def copy_to_local_folder(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, src_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)

def download_file_direct(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def download_files():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_SUBDIR):
        st.info("📥 Téléchargement du modèle depuis Google Drive...")
        tar_path = download_tar_from_drive(MODEL_DRIVE_ID)
        extracted_dir = extract_tar_to_temp(tar_path)
        copy_to_local_folder(extracted_dir, MODEL_SUBDIR)
        st.success("✅ Modèle extrait et copié dans le dossier local.")

    if not os.path.exists(ENCODER_PATH):
        st.info("📥 Téléchargement de l'encodeur...")
        download_file_direct(ENCODER_DRIVE_ID, ENCODER_PATH)
        st.success("✅ Encodeur téléchargé.")

download_files()
def get_arborescence(dossier, indent=0):
    arbo = ""
    try:
        for item in os.listdir(dossier):
            chemin = os.path.join(dossier, item)
            arbo += "  " * indent + "├─ " + item + "\n"
            if os.path.isdir(chemin):
                arbo += get_arborescence(chemin, indent + 1)
    except Exception as e:
        arbo += f"Erreur : {e}\n"
    return arbo





# Affichage dans Streamlit
st.subheader("📁 Arborescence du dossier modèle")
st.text(get_arborescence(MODEL_SUBDIR))

# -------------------------------
# 🔄 Chargement du modèle
# -------------------------------


try:
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_SUBDIR)
    print("✅ Tokenizer chargé avec succès.")
except Exception as e:
    print("❌ Erreur lors du chargement du tokenizer :", e)
model = CamembertForSequenceClassification.from_pretrained(MODEL_SUBDIR)
label_encoder = joblib.load(ENCODER_PATH)
print("✅ encoder chargé avec succès.")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

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
