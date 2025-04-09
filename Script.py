import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Fonction pour le V de Cramér
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

def compute_cramers_v(df, target):
    results = {}
    for col in df.columns:
        if col != target and df[col].dtype == 'object':
            confusion_mat = pd.crosstab(df[col], df[target])
            if confusion_mat.shape[0] > 1 and confusion_mat.shape[1] > 1:
                v = cramers_v(confusion_mat.values)
                results[col] = v
    return pd.Series(results).sort_values(ascending=False)

# Chargement des données
url = "https://raw.githubusercontent.com/JhpAb/Credit-Scoring/main/DATABASE/credit_risk_dataset.csv"
try:
    df = pd.read_csv(url)
    st.success("Fichier CSV chargé avec succès !")
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")
    df = pd.DataFrame()

# Titre
st.title("💳 Credit Scoring Apk")

# Menu latéral
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Aperçu des données",
    "Résumé des données",
    "Traitement des données",
    "Modèle de régression logistique",
    "Scoring des clients",
    "Enregistrement des résultats"
])

# =====================
# Aperçu
if page == "Aperçu des données":
    st.header("Aperçu des données")
    if not df.empty:
        st.write(df.head())
    else:
        st.warning("Données non disponibles.")

# =====================
# Résumé
elif page == "Résumé des données":
    st.header("Résumé des données")
    if not df.empty:
        st.write(df.describe())
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    else:
        st.warning("Données non disponibles.")

# =====================
# Traitement
elif page == "Traitement des données":
    st.header("Traitement des données")
    if not df.empty:
        st.write("Valeurs manquantes :", df.isnull().sum())
        st.write("Nettoyage : suppression des valeurs nulles.")
        df.dropna(inplace=True)
        st.write("Visualisation : boxplot sur les variables numériques.")
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected = st.selectbox("Choisir une variable pour le boxplot :", numeric_cols)
        fig = plt.figure()
        sns.boxplot(data=df[selected])
        st.pyplot(fig)
    else:
        st.warning("Données non disponibles.")

# =====================
# Modélisation
elif page == "Modèle de régression logistique":
    st.header("Modèle de régression logistique")
    if not df.empty:
        target = st.selectbox("Sélectionnez la variable cible :", df.columns)
        
        # Encodage des variables catégorielles
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
        
        st.write("📊 Calcul du V de Cramér pour sélectionner les variables explicatives pertinentes...")
        cramers = compute_cramers_v(df, target)
        top_features = cramers[cramers > 0.1].index.tolist()
        st.write("Variables sélectionnées :", top_features)

        # Multicolinéarité
        corr_matrix = df_encoded[top_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        selected_features = [f for f in top_features if f not in to_drop]
        st.write("Variables retenues après suppression de la multicolinéarité :", selected_features)

        # Modélisation
        X = df_encoded[selected_features]
        y = df_encoded[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("Modèle entraîné avec succès.")
        st.text("Classification report :")
        st.text(classification_report(y_test, y_pred))
        st.text("Matrice de confusion :")
        st.write(confusion_matrix(y_test, y_pred))

        # Sauvegarde du modèle et des colonnes
        st.session_state['model'] = model
        st.session_state['features'] = selected_features
    else:
        st.warning("Données non disponibles.")

# =====================
# Scoring
elif page == "Scoring des clients":
    st.header("Scoring des nouveaux clients")
    if 'model' in st.session_state and 'features' in st.session_state:
        model = st.session_state['model']
        features = st.session_state['features']
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
        X_full = df_encoded[features]
        df['score'] = model.predict_proba(X_full)[:, 1]

        # Attribution des notes S&P
        def score_to_rating(score):
            if score > 0.90: return "AAA"
            elif score > 0.80: return "AA"
            elif score > 0.70: return "A"
            elif score > 0.60: return "BBB"
            elif score > 0.50: return "BB"
            elif score > 0.40: return "B"
            elif score > 0.30: return "CCC"
            elif score > 0.20: return "CC"
            elif score > 0.10: return "C"
            else: return "D"

        df['note'] = df['score'].apply(score_to_rating)
        st.write(df[['score', 'note']].head())
    else:
        st.warning("Veuillez d'abord entraîner le modèle dans la section précédente.")

# =====================
# Enregistrement
elif page == "Enregistrement des résultats":
    st.header("Enregistrement des résultats")
    if 'score' in df.columns and 'note' in df.columns:
        st.write("Prévisualisation des résultats :")
        st.write(df[['score', 'note']].head())
        csv = df.to_csv(index=False)
        st.download_button("Télécharger les résultats", csv, file_name="resultats_clients.csv", mime="text/csv")
    else:
        st.warning("Veuillez effectuer le scoring des clients avant de télécharger les résultats.")

# =====================
# Pied de page
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Auteur :** Votre Nom")
st.sidebar.markdown("📞 **Téléphone :** +225 00000000")
st.sidebar.markdown("📧 **Email :** contact@example.com")
st.sidebar.info("👈 Naviguez dans les sections pour explorer l'application.")
