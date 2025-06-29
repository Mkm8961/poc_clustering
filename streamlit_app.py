import streamlit as st
import pandas as pd

st.set_page_config(page_title="Material Clustering Viewer", layout="wide")
st.title("🔍 Semantic Material Clustering Explorer")

# Directly load the preprocessed Excel file
@st.cache_data
def load_data():
    return pd.read_excel("Material_Cluster_Summary.xlsx")

df = load_data()

# Show entire data on checkbox
if st.checkbox("Show full cluster summary"):
    st.dataframe(df)

# Select cluster to explore
cluster_ids = sorted(df['Cluster_ID'].dropna().unique())
selected_cluster = st.selectbox("Select a Cluster ID", cluster_ids)

# Filter and display members of selected cluster
cluster_df = df[df['Cluster_ID'] == selected_cluster]

st.subheader(f"🧪 Cluster {selected_cluster} Members")
st.write("### 🔢 Clustered Material Descriptions")
st.dataframe(cluster_df[['Cluster_Members_Unique', 'Cluster_MATERIAL_NUMBERS']])

# Show suggested master material number
st.markdown("### ✅ Suggested Representative Material Number")
st.success(cluster_df['Max_MATERIAL_NUMBER'].values[0])
