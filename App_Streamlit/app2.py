"""
Estime ton loyer — Application Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Estime ton loyer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Sur streamlit

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* En-tête principal */
  .hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
  }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    color: #ffffff;
    font-size: 2.8rem;
    margin: 0 0 0.4rem 0;
  }
  .hero p {
    color: #a8d8ea;
    font-size: 1.05rem;
    margin: 0;
  }

  /* Carte résultat */
  .result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-card .loyer-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #4fc3f7;
    line-height: 1;
  }
  .result-card .loyer-label {
    color: #90caf9;
    font-size: 1rem;
    margin-top: 0.3rem;
  }
  .result-card .interval {
    color: #78909c;
    font-size: 0.9rem;
    margin-top: 0.8rem;
  }
  .result-card .m2 {
    background: #0f3460;
    color: #4fc3f7;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    display: inline-block;
    font-size: 0.9rem;
    margin-top: 0.8rem;
  }

  /* Badges métriques */
  .metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
  }
  .metric-box {
    flex: 1;
    background: #f8fafc;
    border-left: 4px solid #4fc3f7;
    border-radius: 8px;
    padding: 0.8rem 1rem;
  }
  .metric-box .label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; }
  .metric-box .value { font-size: 1.3rem; font-weight: 600; color: #1e293b; }

  /* Section headers */
  .section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1e293b;
    border-bottom: 3px solid #4fc3f7;
    padding-bottom: 0.4rem;
    margin: 2rem 0 1rem 0;
  }

  /* Sidebar */
  section[Data-testid="stSidebar"] {
    background-color: #f1f5f9;
  }

  /* Masquer le menu hamburger Streamlit */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# CHARGEMENT DES DONNÉES (mis en cache)

@st.cache_resource
def load_model():
    model    = joblib.load("./src/models/model_loyer.pkl")
    features = json.load(open("./src/models/features.json"))
    return model, features

@st.cache_data
def load_stats():
    """Charge loyers_par_cp.csv — généré par le notebook de nettoyage."""
    df = pd.read_csv("./Data/loyers_par_cp.csv", sep=";")
    df["label"] = df["code_postal"].astype(str) + " – " + df["ville_principale"].fillna("")
    return df

@st.cache_data
def load_annonces():
    """Charge le fichier brut pour les graphiques d'analyse."""
    df = pd.read_csv("./Data/annonces_enrichies_int.csv", sep=";", low_memory=False)
    mapping = {"appartement":"Appartement","APPARTEMENT":"Appartement",
               "studio":"Studio","maison":"Maison"}
    df["type_bien"] = df["type_bien"].str.strip().replace(mapping)
    df = df[(df["loyer"] > 250) & (df["loyer"] <= 6000) &
            (df["surface"] > 5) & (df["surface"] <= 300)]
    return df

# Chargement (avec gestion d'erreur propre)
try:
    model, features = load_model()
    model_ok = True
except Exception as e:
    st.error(f"⚠️ Modèle non trouvé : {e}\n\nLance d'abord `entrainement_modele.ipynb`.")
    model_ok = False

try:
    df_stats = load_stats()
    stats_ok = True
except Exception as e:
    st.warning(f"⚠️ loyers_par_cp.csv non trouvé : {e}")
    stats_ok = False

try:
    df_loyers = load_annonces()
    data_ok = True
except Exception as e:
    st.warning(f"⚠️ annonces_enrichies_int.csv non trouvé : {e}")
    data_ok = False

# FONCTION DE PRÉDICTION

def predict_loyer(surface, nb_pieces, loyer_m2_moyen, code_postal, type_bien="Appartement"):
    type_map = {"Appartement": 0, "Studio": 1, "Maison": 2}
    cp_str   = str(int(code_postal))

    row = {
        "surface":           surface,
        "log_surface":       np.log1p(surface),
        "nb_pieces":         nb_pieces,
        "surface_par_piece": round(surface / nb_pieces, 1) if nb_pieces > 0 else surface,
        "loyer_m2_moyen":    loyer_m2_moyen,
        "code_postal":       code_postal,
        "est_idf":           int(cp_str.startswith(("75","77","78","91","92","93","94","95"))),
        "est_paris":         int(cp_str.startswith("75")),
        "type_encoded":      type_map.get(type_bien, 0),
    }

    X_input = pd.DataFrame([row])[features]
    loyer   = model.predict(X_input)[0]
    margin  = loyer * 0.15

    return {
        "loyer":     round(loyer, 0),
        "bas":       round(loyer - margin, 0),
        "haut":      round(loyer + margin, 0),
        "loyer_m2":  round(loyer / surface, 1),
    }


# EN-TÊTE


st.markdown("""
<div class="hero">
  <h1> Estime ton loyer</h1>
  <p>Estimation du loyer mensuel d'un bien immobilier en France </p>
</div>
""", unsafe_allow_html=True)



# NAVIGATION PAR ONGLETS

tab1, tab2, tab3 = st.tabs([" Estimation personnalisée", "Analyses", " À propos du modèle"])



# TAB 1 — ESTIMATION

with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    # ── Formulaire
    with col_form:
        st.markdown('<div class="section-title">Caractéristiques du bien</div>', unsafe_allow_html=True)

        type_bien = st.selectbox(
            "Type de bien",
            ["Appartement", "Studio", "Maison"],
            help="Le type de bien influence significativement le loyer estimé."
        )

        surface = st.slider("Surface (m²)", min_value=10, max_value=300, value=50, step=1)
        nb_pieces = st.slider("Nombre de pièces", min_value=1, max_value=10, value=2)

        st.markdown("**Localisation**")
        if stats_ok:
            selected_label = st.selectbox(
                "Code postal – Ville",
                df_stats["label"].sort_values(),
                help="Le loyer moyen au m² de la zone sera extrait automatiquement."
            )
            code_postal = int(df_stats.loc[df_stats["label"] == selected_label, "code_postal"].values[0])
            ligne = df_stats[df_stats["code_postal"] == code_postal]
            m2_moyen = float(ligne["loyer_m2_median"].values[0]) if not ligne.empty else None
        else:
            code_postal = st.number_input("Code postal", value=75011, min_value=1000, max_value=99999)
            m2_moyen    = 20.0  # valeur par défaut si loyers_par_cp.csv absent

        btn = st.button(
            "📊 Estimer le loyer",
            type="primary",
            use_container_width=True,
            disabled=not model_ok
        )

    # Résultat
    with col_result:
        st.markdown('<div class="section-title">Résultat de l\'estimation</div>', unsafe_allow_html=True)

        if btn and model_ok and m2_moyen:
            res = predict_loyer(surface, nb_pieces, m2_moyen, code_postal, type_bien)

            # Carte principale
            st.markdown(f"""
            <div class="result-card">
              <div class="loyer-label">Loyer estimé</div>
              <div class="loyer-value">{res['loyer']:,.0f} €</div>
              <div class="loyer-label">par mois</div>
              <div class="interval">Intervalle : {res['bas']:,.0f} € – {res['haut']:,.0f} €</div>
              <div class="m2">{res['loyer_m2']} €/m²</div>
            </div>
            """, unsafe_allow_html=True)

            # Jauge plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["loyer"],
                number={"suffix": " €", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, max(3000, res["haut"] * 1.2)], "ticksuffix": "€"},
                    "bar": {"color": "#4fc3f7"},
                    "steps": [
                        {"range": [0, res["bas"]],  "color": "#e3f2fd"},
                        {"range": [res["bas"], res["haut"]], "color": "#bbdefb"},
                    ],
                    "threshold": {
                        "line": {"color": "#0288d1", "width": 3},
                        "thickness": 0.85,
                        "value": res["loyer"],
                    },
                },
                title={"text": "Position dans l'intervalle", "font": {"size": 13}},
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Détail des inputs utilisés
            with st.expander("🔍 Détail du calcul"):
                st.markdown(f"""
                | Paramètre | Valeur |
                |---|---|
                | Type de bien | {type_bien} |
                | Surface | {surface} m² |
                | Nombre de pièces | {nb_pieces} |
                | Surface/pièce | {surface/nb_pieces:.1f} m² |
                | Code postal | {code_postal} |
                | Loyer moyen zone | {m2_moyen:.1f} €/m² |
                | Zone IDF | {" Oui" if str(code_postal).startswith(("75","77","78","91","92","93","94","95")) else "Non"} |
                | Paris | {"Oui" if str(code_postal).startswith("75") else " Non"} |
                """)

        elif btn and not m2_moyen:
            st.error("Code postal non reconnu dans les données.")

        else:
            st.markdown("""
            <div style="text-align:center; color:#94a3b8; padding:3rem 1rem;">
              <div style="font-size:3rem;"></div>
              <div style="margin-top:0.5rem;">
                Remplis le formulaire et clique sur<br><b>Estimer le loyer</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Exemples rapides
            st.markdown("**Exemples rapides :**")
            exemples = [
                ("Studio Paris 11e", 22, 1, 75011, "Studio"),
                ("Appart Lyon 3e",   45, 2, 69003, "Appartement"),
                (" Maison Nantes",    90, 4, 44000, "Maison"),
            ]
            for label, surf, pieces, cp, tb in exemples:
                if st.button(label, use_container_width=True):
                    if stats_ok:
                        l = df_stats[df_stats["code_postal"] == cp]
                        m2 = float(l["loyer_m2_median"].values[0]) if not l.empty else 20.0
                    else:
                        m2 = 20.0
                    if model_ok:
                        r = predict_loyer(surf, pieces, m2, cp, tb)
                        st.success(f"**{label}** → ~{r['loyer']:,.0f} €/mois")



# ANALYSES

with tab2:

    st.markdown('<div class="section-title">Panorama des loyers en France</div>', unsafe_allow_html=True)

    # ── Filtres
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        types_choisis = st.multiselect(
            "Type de bien",
            ["Appartement", "Studio", "Maison"],
            default=["Appartement", "Studio"],
        )
    with col_f2:
        plage_loyer = st.slider("Plage de loyer (€)", 250, 6000, (250, 3000), step=50)

    df_viz = df_loyers[
        (df_loyers["type_bien"].isin(types_choisis)) &
        (df_loyers["loyer"].between(*plage_loyer))
    ]
    st.caption(f"{len(df_viz):,} annonces affichées")

    # ── Ligne 1 : distribution + pie
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_hist = px.histogram(
            df_viz, x="loyer", nbins=80,
            color="type_bien",
            color_discrete_map={"Appartement":"#4fc3f7","Studio":"#81c784","Maison":"#ffb74d"},
            labels={"loyer": "Loyer (€/mois)", "count": "Nb annonces"},
            title="Distribution des loyers",
            opacity=0.8,
        )
        fig_hist.update_layout(
            bargap=0.05, legend_title="Type",
            margin=dict(t=40, b=20, l=20, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        tranches = pd.cut(df_viz["loyer"],
                          bins=[0, 500, 800, 1200, 1800, 3000, 20000],
                          labels=["< 500€", "500–800€", "800–1 200€",
                                  "1 200–1 800€", "1 800–3 000€", "> 3 000€"])
        fig_pie = px.pie(
            names=tranches.value_counts().index,
            values=tranches.value_counts().values,
            title="Répartition par tranche",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Ligne 2 : top villes + loyer vs surface
    col3, col4 = st.columns([1, 1])

    with col3:
        if "ville" in df_viz.columns:
            top_villes = (
                df_viz[df_viz["ville"].notna()]
                .groupby("ville")
                .agg(nb=("loyer","count"), med=("loyer_m2_moyen","median"))
                .query("nb >= 30")
                .sort_values("med", ascending=False)
                .head(15)
                .reset_index()
            )
            fig_bar = px.bar(
                top_villes, x="med", y="ville",
                orientation="h",
                text="med",
                color="med",
                color_continuous_scale="Blues",
                labels={"med": "Loyer/m² médian (€)", "ville": ""},
                title="Top 15 villes — loyer médian au m²",
            )
            fig_bar.update_traces(texttemplate="%{text:.0f} €", textposition="outside")
            fig_bar.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=60),
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        df_sc = df_viz[(df_viz["surface"] <= 150)].sample(min(3000, len(df_viz)), random_state=42)
        fig_scatter = px.scatter(
            df_sc, x="surface", y="loyer",
            color="type_bien",
            color_discrete_map={"Appartement":"#4fc3f7","Studio":"#81c784","Maison":"#ffb74d"},
            opacity=0.3,
            trendline="ols",
            labels={"surface": "Surface (m²)", "loyer": "Loyer (€)"},
            title="Loyer vs Surface",
        )
        fig_scatter.update_traces(marker={"size": 4})
        fig_scatter.update_layout(
            margin=dict(t=40, b=20, l=20, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            legend_title="Type",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Ligne 3 : boxplot loyers par source
    if "web_source" in df_viz.columns:
        fig_box = px.box(
            df_viz, x="web_source", y="loyer",
            color="web_source",
            color_discrete_sequence=["#4fc3f7", "#81c784", "#ffb74d"],
            labels={"web_source": "Source", "loyer": "Loyer (€/mois)"},
            title="Distribution des loyers par source de données",
        )
        fig_box.update_layout(
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_box, use_container_width=True)


# TAB 3 — À PROPOS DU MODÈLE
with tab3:
    st.markdown('<div class="section-title">À propos du modèle</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🌲 Random Forest Regressor")
        st.markdown("""
        | Paramètre | Valeur |
        |---|---|
        | Algorithme | Random Forest |
        | n_estimators | 300 arbres |
        | max_depth | 20 |
        | min_samples_leaf | 5 |
        | max_features | sqrt |
        """)

        # Charger les métriques si dispo
        try:
            with open("./models/metriques.json") as f:
                m = json.load(f)
            st.markdown("####  Performances sur le jeu de test")
            cols = st.columns(4)
            cols[0].metric("MAE",  f"{m['mae']:.0f} €")
            cols[1].metric("RMSE", f"{m['rmse']:.0f} €")
            cols[2].metric("R²",   f"{m['r2']:.3f}")
            cols[3].metric("MAPE", f"{m['mape']:.1f} %")
            st.caption(f"Entraîné sur {m['nb_train']:,} annonces · testé sur {m['nb_test']:,}")
        except FileNotFoundError:
            st.info("Lance `entrainement_modele.ipynb` pour générer les métriques.")

    with col_b:
        st.markdown("####  Features utilisées")
        features_info = {
            "surface":           "Surface du bien (m²)",
            "log_surface":       "Log de la surface (capture la non-linéarité)",
            "nb_pieces":         "Nombre de pièces",
            "surface_par_piece": "Surface moyenne par pièce",
            "loyer_m2_moyen":    "Prix moyen au m² dans la zone ",
            "code_postal":       "Localisation brute",
            "est_idf":           "Bien en Île-de-France (0/1)",
            "est_paris":         "Bien à Paris 75xxx (0/1)",
            "type_encoded":      "Type de bien encodé (Appt/Studio/Maison)",
        }
        for feat, desc in features_info.items():
            st.markdown(f"- **`{feat}`** — {desc}")

    st.markdown("---")
    st.markdown("####  Données source")
    st.markdown("""
    | Source | Description |
    |---|---|
    | SeLoger | 56 780 annonces scrappées |
    | Leboncoin | 26 853 annonces |
    | BienIci | 11 569 annonces |
    | **Total après nettoyage** | **~90 971 annonces** |
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding: 1rem 0;">
      Estime ton loyer · Projet Data Science · 2025<br>
      Random Forest · scikit-learn · Streamlit · Plotly
    </div>
    """, unsafe_allow_html=True)
