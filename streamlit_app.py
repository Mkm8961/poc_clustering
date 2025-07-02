# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import umap
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import streamlit as st

# -----------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------
def preprocess_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    replacements = {
        r'\bin\b': 'inch',
        r'\bdia\b': 'diameter',
        r'\blg\b': 'length',
        r'\bwd\b': 'width',
        r'\bid\b': 'inner_diameter',
        r'\bod\b': 'outer_diameter',
        r'\bno\b': 'normally_open',
        r'\bnc\b': 'normally_closed',
        r'\s+': ' '
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text.strip()

# -----------------------------
# SAP MM VALIDATION FUNCTION
# -----------------------------
def validate_sap_mm_compliance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Short_Text_Valid'] = df['MATERIAL_NUMBER_TEXT'].apply(lambda x: len(str(x)) <= 40)
    df['Truncated_Short_Text'] = df['MATERIAL_NUMBER_TEXT'].apply(lambda x: str(x)[:40])
    df['LONG_TEXT'] = df['Cleaned_Text'].str.upper()
    df['Invalid_Characters'] = df['MATERIAL_NUMBER_TEXT'].apply(lambda x: bool(re.search(r"[-/#@(){}\[\]*\"?<>\^~|]", str(x))))
    df['Format_OK'] = df['Cleaned_Text'].apply(lambda text: len(text.split()) >= 3 and any(unit in text for unit in ['inch', 'mm', 'ft', 'meter']))
    df['May_Contain_Manufacturer_Info'] = df['Cleaned_Text'].apply(
        lambda text: bool(re.search(r'\b(model|pn|part|no|serial|sn|mfg|ref)\b', str(text).lower())) or bool(re.search(r'[A-Z]{2,}\d+', str(text)))
    )
    return df

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def embed_with_sbert(df: pd.DataFrame, text_column='Cleaned_Text', model_name='all-MiniLM-L6-v2') -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_column].tolist(), show_progress_bar=True, batch_size=32)
    return embeddings

# -----------------------------
# CLUSTERING FUNCTION
# -----------------------------
def cluster_descriptions(embeddings: np.ndarray, eps=0.01, min_samples=2) -> np.ndarray:
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = np.clip(1 - similarity_matrix, 0, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    return clustering.fit_predict(distance_matrix)

# -----------------------------
# EVALUATE CLUSTERS
# -----------------------------
def evaluate_clusters(embeddings: np.ndarray, labels: np.ndarray):
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        score = silhouette_score(embeddings[mask], labels[mask])
        st.info(f"Silhouette Score: {score:.3f}")
    else:
        st.warning("Not enough valid clusters to compute silhouette score.")

# -----------------------------
# CLUSTER SUMMARY
# -----------------------------
def create_cluster_summary(df: pd.DataFrame, cluster_col='Cluster_SBERT') -> pd.DataFrame:
    cluster_rows = []
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            continue  # Skip noise
        group = df[df[cluster_col] == cluster_id]
        text_members = group['MATERIAL_NUMBER_TEXT'].astype(str).tolist()
        unique_text_members = group['MATERIAL_NUMBER_TEXT'].astype(str).unique().tolist()
        material_numbers = group['MATERIAL_NUMBER_1'].astype(str).unique().tolist()
        try:
            max_material_number = max(
                material_numbers,
                key=lambda x: float(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else -1
            )
        except:
            max_material_number = material_numbers[0] if material_numbers else ""

        cluster_rows.append({
            'Cluster_ID': cluster_id,
            'Cluster_Members_All': ' | '.join(text_members),
            'Cluster_Members_Unique': ' | '.join(unique_text_members),
            'Cluster_MATERIAL_NUMBERS': ' | '.join(material_numbers),
            'Max_MATERIAL_NUMBER': max_material_number
        })

    return pd.DataFrame(cluster_rows)

# -----------------------------
# UMAP PLOT RETURN AS FIGURE
# -----------------------------
def get_umap_plot(embeddings: np.ndarray, labels: np.ndarray, title="UMAP Projection of Clusters"):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=labels, cmap='Spectral', s=10, alpha=0.7
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster ID")
    return fig

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="Material Clustering App", layout="wide")
st.title("🔍 Semantic Material Clustering App")

st.markdown("Upload a CSV with columns `MATERIAL_NUMBER_TEXT` and `MATERIAL_NUMBER_1` to begin.")

uploaded_file = st.file_uploader("📁 Upload Material CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df[['MATERIAL_NUMBER_TEXT', 'MATERIAL_NUMBER_1']].dropna(how='all')
    df['Cleaned_Text'] = df['MATERIAL_NUMBER_TEXT'].apply(preprocess_text)
    df = validate_sap_mm_compliance(df)

    st.info("Embedding and clustering may take a minute...")

    with st.spinner("🔁 Embedding with SBERT..."):
        embeddings = embed_with_sbert(df)

    with st.spinner("🔁 Clustering using DBSCAN..."):
        df['Cluster_SBERT'] = cluster_descriptions(embeddings)

    with st.spinner("📊 Evaluating Clusters..."):
        evaluate_clusters(embeddings, df['Cluster_SBERT'].values)

    with st.spinner("📊 Creating Summary..."):
        summary_df = create_cluster_summary(df)

    st.success("✅ Clustering complete!")

    if st.checkbox("📋 Show Cluster Summary"):
        st.dataframe(summary_df)

    selected_cluster = st.selectbox("🔢 Explore a Cluster", summary_df['Cluster_ID'].unique())

    cluster_row = summary_df[summary_df['Cluster_ID'] == selected_cluster].iloc[0]
    st.markdown(f"### Cluster {selected_cluster} Details")
    st.write(f"**Material Numbers:** {cluster_row['Cluster_MATERIAL_NUMBERS']}")
    st.write(f"**Representative Number:** {cluster_row['Max_MATERIAL_NUMBER']}")
    st.write(f"**Descriptions:**")
    st.text_area("Descriptions", cluster_row['Cluster_Members_Unique'], height=150)

    feedback = st.text_input("📝 Suggest improvements (optional)")
    if st.button("Submit Feedback"):
        with open("cluster_feedback.txt", "a") as f:
            f.write(f"Cluster {selected_cluster}: {feedback}\n")
        st.success("✅ Feedback submitted")

    st.markdown("### 📉 UMAP Visualization")
    fig = get_umap_plot(embeddings, df['Cluster_SBERT'].values)
    st.pyplot(fig)

    st.markdown("### 📥 Download Cluster Summary")
    output = BytesIO()
    summary_df.to_excel(output, index=False)
    st.download_button("📤 Download Excel", data=output.getvalue(), file_name="Material_Cluster_Summary.xlsx")
