import os
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score, ndcg_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_movie_data():
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')
    merged_df = pd.merge(ratings_df, movies_df, on='movieId')
    merged_df['genres'] = merged_df['genres'].str.split('|')
    return merged_df

def process_data(df):
    movie_scores = {}
    user_id = df['userId']
    movie_id = df['movieId']
    rating = df['rating']
    genres = df['genres']
    for i in range(len(user_id)):
        uid = user_id[i]
        mid = movie_id[i]
        r = rating[i]
        if mid not in movie_scores:
            movie_scores[mid] = {'total_score': 0, 'count': 0}
        movie_scores[mid]['total_score'] += r
        movie_scores[mid]['count'] += 1
    num_unique_user_ids = df['userId'].nunique()
    num_unique_movie_ids = df['movieId'].nunique()
    num_unique_genres = len(set([g for sublist in df['genres'] for g in sublist]))
    num_interactions = df[['userId', 'movieId']].drop_duplicates().shape[0]
    sparsity = (1 - (num_interactions / (num_unique_user_ids * num_unique_movie_ids))) * 100
    user_rating_counts = df.groupby('userId')['movieId'].nunique()
    user_count_histogram = user_rating_counts.value_counts().sort_index()
    for count, users in user_count_histogram.items():
        print(f" [{count}] {users} users have {count} rated movies")

def Load_Into_Graph(df):
    logging.info("Loading movie data into a graph...")
    G = nx.Graph()
    all_genres = set()
    for genres in df['genres']:
        all_genres.update(genres)
    G.add_nodes_from(all_genres, node_type='genre')
    for _, row in df.iterrows():
        uid = f"u{row['userId']}"
        mid = f"m{row['movieId']}"
        G.add_node(uid, node_type='user')
        G.add_node(mid, node_type='movie')
        G.add_edge(uid, mid, weight=row['rating'], edge_type='rating')
        for genre in row['genres']:
            G.add_edge(mid, genre, edge_type='genre')
    logging.info("Finished building movie graph")
    logging.info(f"Graph contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def find_paths_users_interests(df):
    G = Load_Into_Graph(df)
    df['avg_rating'] = df.groupby('movieId')['rating'].transform('mean')
    meta_path = ['user', 'movie', 'genre']
    logging.info(f"Meta-Path: {' -> '.join(meta_path)}")
    paths = []
    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            user_rated_movies = [mid for mid in G.neighbors(uid) if G.nodes[mid]['node_type'] == 'movie']
            for mid in user_rated_movies:
                movie_id = int(mid[1:])
                movie_rows = df[df['movieId'] == movie_id]
                if not movie_rows.empty:
                    avg_rating = movie_rows['avg_rating'].iloc[0]
                    actual_rating = movie_rows[movie_rows['userId'] == int(uid[1:])]['rating'].iloc[0]
                    if actual_rating >= avg_rating:
                        genre_nodes = [node for node in G.neighbors(mid) if G.nodes[node]['node_type'] == 'genre']
                        for genre in genre_nodes:
                            paths.append([uid, mid, genre])
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    genre_encoder = LabelEncoder()
    user_encoder.fit([path[0] for path in paths])
    movie_encoder.fit([path[1] for path in paths])
    genre_encoder.fit([path[2] for path in paths])
    encoded_paths = [
        [user_encoder.transform([path[0]])[0],
         movie_encoder.transform([path[1]])[0],
         genre_encoder.transform([path[2]])[0]]
        for path in paths
    ]
    paths_tensor = torch.tensor(encoded_paths, dtype=torch.long)
    return paths_tensor, meta_path

class NLA(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim, paths):
        super(NLA, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.paths = paths.clone().detach() if paths is not None else None

    def forward(self, uid, mid, gid):
        user_emb = self.user_embedding(uid)
        movie_emb = self.movie_embedding(mid)
        genre_emb = self.genre_embedding(gid)
        if self.paths is not None:
            max_size = max(user_emb.size(0), movie_emb.size(0), genre_emb.size(0))
            user_emb = F.pad(user_emb, (0, 0, 0, max_size - user_emb.size(0)))
            movie_emb = F.pad(movie_emb, (0, 0, 0, max_size - movie_emb.size(0)))
            genre_emb = F.pad(genre_emb, (0, 0, 0, max_size - genre_emb.size(0)))
            user_similarity = torch.matmul(user_emb, user_emb.t())
            movie_similarity = torch.matmul(movie_emb, movie_emb.t())
            genre_similarity = torch.matmul(genre_emb, genre_emb.t())
            user_attention = F.softmax(user_similarity, dim=-1)
            movie_attention = F.softmax(movie_similarity, dim=-1)
            genre_attention = F.softmax(genre_similarity, dim=-1)
            user_weighted_emb = torch.matmul(user_attention, user_emb)
            movie_weighted_emb = torch.matmul(movie_attention, movie_emb)
            genre_weighted_emb = torch.matmul(genre_attention, genre_emb)
            node_embeddings = torch.cat((user_weighted_emb, movie_weighted_emb, genre_weighted_emb), dim=1)
        else:
            node_embeddings = torch.cat((user_emb, movie_emb, genre_emb), dim=1)
        return node_embeddings

    def train_nla(self, df, user_encoder, movie_encoder, genre_encoder, num_epochs, save_path='nla_model.parameters'):
        criterion_nla = nn.MSELoss()
        optimizer_nla = optim.Adam(self.parameters(), lr=0.01)
        dataset = HeterogeneousDataset(df, user_encoder, movie_encoder, genre_encoder)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.to('cuda')
        best_loss = float('inf')
        for epoch in range(num_epochs):
            running_loss_nla = 0.0
            for uid, mid, gid, label in data_loader:
                optimizer_nla.zero_grad()
                uid, mid, gid, label = uid.cuda(), mid.cuda(), gid.cuda(), label.cuda()
                embeddings = self(uid, mid, gid)
                label = label.unsqueeze(1).float()
                label = label.repeat(1, embeddings.size(1))
                loss_nla = criterion_nla(embeddings, label)
                running_loss_nla += loss_nla.item()
                loss_nla.backward()
                optimizer_nla.step()
            avg_loss_nla = running_loss_nla / len(data_loader)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, NLA Loss: {avg_loss_nla:.4f}")
            if avg_loss_nla < best_loss:
                best_loss = avg_loss_nla
                torch.save(self.state_dict(), save_path)
                logging.info(f"Model saved at epoch {epoch + 1} with loss {avg_loss_nla:.4f}")
        return avg_loss_nla

    def get_embeddings(self, uid, mid, gid):
        with torch.no_grad():
            embeddings = self(uid, mid, gid)
        return embeddings

class HeterogeneousDataset(Dataset):
    def __init__(self, df, user_encoder, movie_encoder, genre_encoder):
        self.uids = user_encoder.transform(df['userId'].astype(str).apply(lambda x: f"u{x}"))
        self.mids = movie_encoder.transform(df['movieId'].astype(str).apply(lambda x: f"m{x}"))
        self.gids = genre_encoder.transform(df['genres'].apply(lambda x: x[0] if x else 'Unknown'))
        self.labels = df['rating'].astype(float).values

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        return (
            self.uids[idx],
            self.mids[idx],
            self.gids[idx],
            self.labels[idx]
        )

class ContrastiveModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim):
        super(ContrastiveModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.temperature = 0.07

    def forward(self, uid, mid, gid):
        user_emb = self.user_embedding(uid)
        movie_emb = self.movie_embedding(mid)
        genre_emb = self.genre_embedding(gid)
        return user_emb, movie_emb, genre_emb

    def contrastive_loss(self, anchor_emb, positive_emb, negative_emb):
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1) / self.temperature
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=1) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor_emb.device)
        loss = F.cross_entropy(logits, labels)
        return loss

class ContrastiveDataset(Dataset):
    def __init__(self, df, user_encoder, movie_encoder, genre_encoder):
        self.df = df
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        self.genre_encoder = genre_encoder
        self.user_movie_map = {}
        for user_id, group in df.groupby('userId'):
            self.user_movie_map[user_id] = {
                'positive': group[group['rating'] > 3]['movieId'].tolist(),
                'negative': group[group['rating'] <= 3]['movieId'].tolist()
            }
        self.samples = []
        for user_id, movies in self.user_movie_map.items():
            if movies['positive'] and movies['negative']:
                for pos_movie in movies['positive']:
                    neg_movie = random.choice(movies['negative'])
                    self.samples.append((user_id, pos_movie, neg_movie))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, pos_movie, neg_movie = self.samples[idx]
        user_genres = self.df[self.df['userId'] == user_id]['genres'].iloc[0][0] if self.df[self.df['userId'] == user_id]['genres'].iloc[0] else 'Unknown'
        pos_genres = self.df[self.df['movieId'] == pos_movie]['genres'].iloc[0][0] if self.df[self.df['movieId'] == pos_movie]['genres'].iloc[0] else 'Unknown'
        neg_genres = self.df[self.df['movieId'] == neg_movie]['genres'].iloc[0][0] if self.df[self.df['movieId'] == neg_movie]['genres'].iloc[0] else 'Unknown'
        uid = self.user_encoder.transform([f"u{user_id}"])[0]
        pos_mid = self.movie_encoder.transform([f"m{pos_movie}"])[0]
        neg_mid = self.movie_encoder.transform([f"m{neg_movie}"])[0]
        user_gid = self.genre_encoder.transform([user_genres])[0]
        pos_gid = self.genre_encoder.transform([pos_genres])[0]
        neg_gid = self.genre_encoder.transform([neg_genres])[0]
        return uid, pos_mid, neg_mid, user_gid, pos_gid, neg_gid

def normalize_summed_embeddings(embeddings):
    embeddings = embeddings.to('cpu').detach().numpy()
    scaler = MinMaxScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    return normalized_embeddings

def process_user(i, df, normalized_embeddings, similarity_threshold, AUC_popular):
    user_id = df.iloc[i]['userId']
    similarities = cosine_similarity([normalized_embeddings[i]], normalized_embeddings)[0]
    similar_users = [
        (j, similarity_score) for j, similarity_score in enumerate(similarities)
        if j != i and similarity_score >= similarity_threshold
    ]
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    actual_ratings = set(df[df['userId'] == user_id]['movieId'])
    similar_user_rated_movies = set()
    for similar_user_index, _ in similar_users:
        similar_user_id = df.iloc[similar_user_index]['userId']
        if similar_user_id != user_id:
            similar_user_rated_movies.update(df[df['userId'] == similar_user_id]['movieId'])
    movie_popularity = {}
    for movie_id in similar_user_rated_movies:
        movie_popularity[movie_id] = len(df[df['movieId'] == movie_id])
    popular_movies = sorted(movie_popularity.items(), key=lambda x: x[1], reverse=True)[:AUC_popular]
    recommended_movies = [movie_id for movie_id, _ in popular_movies]
    return user_id, recommended_movies, actual_ratings

def extract_recommendations_and_ratings(df, normalized_embeddings, similarity_threshold, AUC_popular):
    recommendations = {}
    actual_ratings = {}
    num_cores = 8
    results = Parallel(n_jobs=num_cores)(
        delayed(process_user)(i, df, normalized_embeddings, similarity_threshold, AUC_popular)
        for i in range(min(len(df), len(normalized_embeddings)))
    )
    for user_id, recommended_movies, actual_rating in results:
        recommendations[user_id] = recommended_movies
        actual_ratings[user_id] = actual_rating
    return recommendations, actual_ratings

def calculate_Recommendation_AUC(predictions, actual_ratings, test_size=0.2):
    true_labels = []
    predicted_scores = []
    for user_id, recommended_movies in predictions.items():
        rated_movies = actual_ratings.get(user_id, [])
        if recommended_movies:
            train_set, test_set = train_test_split(recommended_movies, test_size=test_size, random_state=42)
            for movie_id in test_set:
                if movie_id in rated_movies:
                    true_labels.append(1)
                else:
                    true_labels.append(0)
                predicted_scores.append(len(recommended_movies) - recommended_movies.index(movie_id))
    auc = roc_auc_score(true_labels, predicted_scores)
    return auc

def calculate_ndcg(df, user_id, recommendations, k):
    user_ratings = df[(df['userId'] == user_id) & df['movieId'].isin(recommendations)][['movieId', 'rating']]
    ratings_dict = dict(zip(user_ratings['movieId'], user_ratings['rating']))
    y_true = [1 if movie_id in ratings_dict and ratings_dict[movie_id] > 0 else 0 for movie_id in recommendations]
    y_true = np.array(y_true).reshape(1, -1)
    y_score = np.array(y_true)
    ndcg = ndcg_score(y_true, y_score, k=k)
    return ndcg

def main():
    df = load_movie_data()
    process_data(df)
    paths_tensor, meta_path = find_paths_users_interests(df)
    logging.info(f"Generated {len(paths_tensor)} meta-paths")
    num_users = df['userId'].nunique()
    num_movies = df['movieId'].nunique()
    num_genres = len(set([g for sublist in df['genres'] for g in sublist]))
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    genre_encoder = LabelEncoder()
    user_encoder.fit([f"u{uid}" for uid in df['userId'].unique()])
    movie_encoder.fit([f"m{mid}" for mid in df['movieId'].unique()])
    genre_encoder.fit(list(set([g for sublist in df['genres'] for g in sublist])))
    embedding_dim = 64
    nla_model = NLA(num_users, num_movies, num_genres, embedding_dim, paths_tensor)
    nla_model.train_nla(df, user_encoder, movie_encoder, genre_encoder, num_epochs=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uid_tensor = torch.LongTensor(list(range(num_users))).to(device)
    mid_tensor = torch.LongTensor(list(range(num_movies))).to(device)
    gid_tensor = torch.LongTensor(list(range(num_genres))).to(device)
    embeddings_nla = nla_model.get_embeddings(uid_tensor, mid_tensor, gid_tensor)
    logging.info("NLA Embeddings generated")
    contrastive_model = ContrastiveModel(num_users, num_movies, num_genres, embedding_dim).to(device)
    contrastive_optimizer = optim.Adam(contrastive_model.parameters(), lr=0.001)
    contrastive_dataset = ContrastiveDataset(df, user_encoder, movie_encoder, genre_encoder)
    contrastive_loader = DataLoader(contrastive_dataset, batch_size=128, shuffle=True)
    logging.info("Training contrastive model...")
    for epoch in range(50):
        total_loss = 0.0
        for batch in contrastive_loader:
            uid, pos_mid, neg_mid, user_gid, pos_gid, neg_gid = [x.to(device) for x in batch]
            contrastive_optimizer.zero_grad()
            user_emb, _, _ = contrastive_model(uid, pos_mid, user_gid)
            pos_movie_emb, _, _ = contrastive_model(pos_mid, pos_mid, pos_gid)
            neg_movie_emb, _, _ = contrastive_model(neg_mid, neg_mid, neg_gid)
            loss = contrastive_model.contrastive_loss(user_emb, pos_movie_emb, neg_movie_emb)
            total_loss += loss.item()
            loss.backward()
            contrastive_optimizer.step()
        avg_loss = total_loss / len(contrastive_loader)
        logging.info(f"Epoch {epoch + 1}/50, Contrastive Loss: {avg_loss:.4f}")
    with torch.no_grad():
        user_emb_contrastive, movie_emb_contrastive, _ = contrastive_model(uid_tensor, mid_tensor, gid_tensor)
    fused_user_emb = embeddings_nla[:, :embedding_dim] + user_emb_contrastive
    fused_movie_emb = embeddings_nla[:, embedding_dim:2 * embedding_dim] + movie_emb_contrastive
    fused_embeddings = torch.cat([fused_user_emb, fused_movie_emb], dim=0)
    normalized_embeddings = normalize_summed_embeddings(fused_embeddings)
    similarity_threshold = 0.7
    AUC_popular = 10
    recommendations, actual_ratings = extract_recommendations_and_ratings(
        df, normalized_embeddings, similarity_threshold, AUC_popular
    )
    test_size = 0.2
    auc_score = calculate_Recommendation_AUC(recommendations, actual_ratings, test_size)
    logging.info(f"Recommendation AUC Score: {auc_score:.4f}")
