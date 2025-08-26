#!/bin/bash
# Script pour lancer Streamlit sur Render de manière stable

# Installer les dépendances
pip install -r requirements.txt

# Lancer Streamlit
streamlit run app.py \
    --server.port $PORT \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
