from data_preprocessor import DataPreprocessor
from feature_extractor import TextDataset, FeatureExtractor
from text_cluster import TextCluster
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

def main():
    # Data Preprocessing
    preprocessor = DataPreprocessor('your_file.csv')
    df = preprocessor.load_data()

    # Feature Extraction
    dataset = TextDataset(texts=df['text'].tolist())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    extractor = FeatureExtractor()
    trainer = pl.Trainer(gpus=1, logger=False)  # Adjust `gpus` as per your setup
    features = extractor.extract_features(dataloader)

    # Clustering
    clusterer = TextCluster(n_clusters=5)
    labels = clusterer.fit_predict(features)

    # Save clustered data
    df['cluster'] = labels
    for cluster in range(5):
        cluster_df = df[df['cluster'] == cluster]
        cluster_df.to_csv(f'cluster_{cluster}.csv', index=False)

if __name__ == '__main__':
    main()
