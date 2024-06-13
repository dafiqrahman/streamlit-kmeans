import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import base64


def to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')


st.title("K-Means Clustering Application")
st.subheader("Ardelia Parahita Arisanti 24050121140166")
# Bagian 1: Input Data
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
separator = st.sidebar.selectbox("Select separator", (",", ";", "\t"))

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=separator)

    # Mengisi nilai yang hilang
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    st.sidebar.write("Columns in dataset:")
    columns = st.sidebar.multiselect(
        "Select columns to use for clustering", data.columns.tolist(), default=data.columns.tolist())
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

    if columns:
        data_selected = data[columns]

        # Encoding data kategorikal
        data_selected = pd.get_dummies(data_selected, drop_first=True)

        # Standardizing the data
        scaler = StandardScaler()
        data_selected_scaled = scaler.fit_transform(data_selected)

        tabs = st.tabs(["Preview Data", "EDA", "Modeling", "Profiling"])

        with tabs[0]:
            # Bagian 2: Preview Data
            st.header("Preview Data")
            st.write(data_selected.describe())
            st.header("Raw Data")
            st.write(data_selected.head())

        with tabs[1]:
            # Bagian 3: EDA
            st.header("Exploratory Data Analysis (EDA)")

            if data_selected.shape[1] > 2:
                pca = PCA(2)
                pca_data = pca.fit_transform(data_selected_scaled)
                pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
                fig = px.scatter(pca_df, x='PCA1', y='PCA2',
                                 title="Scatter Plot of PCA Components")
                st.plotly_chart(fig)
            else:
                fig = px.scatter(pd.DataFrame(data_selected_scaled, columns=data_selected.columns),
                                 x=data_selected.columns[0], y=data_selected.columns[1], title="Scatter Plot")
                st.plotly_chart(fig)

            st.subheader("Distribution of Columns")
            for col in data_selected.columns:
                fig = px.histogram(
                    data_selected, x=col, title=f"Distribution of {col}", marginal="violin", nbins=30)
                st.plotly_chart(fig)

        with tabs[2]:
            # Bagian 4: Pemodelan
            st.header("Modeling")
            silhouette_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data_selected_scaled)
                labels = kmeans.labels_
                score = silhouette_score(data_selected_scaled, labels)
                silhouette_scores.append(score)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(2, 11)),
                          y=silhouette_scores, mode='lines+markers'))
            fig.update_layout(title="Silhouette Scores for K-Means Clustering",
                              xaxis_title="Number of clusters",
                              yaxis_title="Silhouette Score")
            st.plotly_chart(fig)

            # Plotting final clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data_selected['Cluster'] = kmeans.fit_predict(data_selected_scaled)

            if data_selected.shape[1] > 2:
                pca_data = PCA(2).fit_transform(data_selected_scaled)
                pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
                pca_df['Cluster'] = data_selected['Cluster']
                fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Cluster',
                                 title=f"K-Means Clustering with {n_clusters} Clusters")
                st.plotly_chart(fig)
            else:
                fig = px.scatter(pd.DataFrame(data_selected_scaled, columns=data_selected.columns),
                                 x=data_selected.columns[0], y=data_selected.columns[1], color=data_selected['Cluster'], title=f"K-Means Clustering with {n_clusters} Clusters")
                st.plotly_chart(fig)

        with tabs[3]:
            # Bagian 5: Profiling
            st.header("Profiling")
            data_selected['Cluster'] = kmeans.labels_
            centroids_df = data_selected.groupby(
                'Cluster').mean().reset_index()
            # plotting data using bar chart, x is the cluster, y is the mean value of each column
            fig = go.Figure()
            for col in centroids_df.columns:
                if col != 'Cluster':
                    fig.add_trace(
                        go.Bar(x=centroids_df['Cluster'], y=centroids_df[col], name=col))
            fig.update_layout(title="Cluster Profiling",
                              xaxis_title="Cluster",
                              yaxis_title="Mean Value")
            st.plotly_chart(fig)
            st.write(centroids_df)
            # add button to dowload centroids_df
            csv = to_csv(data_selected)
            st.download_button(
                label="Download Cluster Profiling",
                data=csv,
                file_name='cluster_profiling.csv',
                mime='text/csv'
            )

    else:
        st.write("Please select at least one column for clustering")
