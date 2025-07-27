import os
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import logging
import random

HEALTH_THRESHOLD = 0.7

WHO_STANDARDS = {
    'fibers': lambda x: x > 10,
    'fat': lambda x: 15 <= x <= 30,
    'sugar': lambda x: x < 10,
    'sodium': lambda x: x < 5,
    'protein': lambda x: 10 <= x <= 15,
    'saturated_fat': lambda x: x < 10,
    'carbohydrates': lambda x: 55 <= x <= 75
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_health_score(nutrition_data):
    nutrients = {
        'fibers': nutrition_data[0],
        'fat': nutrition_data[1],
        'sugar': nutrition_data[2],
        'sodium': nutrition_data[3],
        'protein': nutrition_data[4],
        'saturated_fat': nutrition_data[5],
        'carbohydrates': nutrition_data[6]
    }
    conditions_met = 0
    for nutrient_name, value in nutrients.items():
        if nutrient_name in WHO_STANDARDS and WHO_STANDARDS[nutrient_name](value):
            conditions_met += 1
    health_score = conditions_met / 7.0
    return health_score

def is_healthy(nutrition_data):
    health_score = calculate_health_score(nutrition_data)
    return health_score >= HEALTH_THRESHOLD

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['health_score'] = df['nutrition'].apply(calculate_health_score)
    df['is_healthy'] = df['health_score'].apply(lambda x: 1 if x >= HEALTH_THRESHOLD else 0)
    return df

def build_food_graph(df):
    G = nx.Graph()
    nutrients = ["Proteins", "Carbohydrates", "Sugars", "Sodium", "Fat", "Saturated_fats", "Fibers"]
    G.add_nodes_from(nutrients, node_type='nutrition')
    health_status = {}
    for _, row in df.iterrows():
        uid = row['user_id']
        rid = row['recipe_id']
        rating = row['rating']
        ingredients = row['ingredients']
        nutrition = row['nutrition']
        health_score = row['health_score']
        user_node = f"u{uid}"
        recipe_node = f"r{rid}"
        G.add_node(user_node, node_type='user')
        G.add_node(recipe_node, node_type='recipe', health_score=health_score)
        health_status[recipe_node] = health_score
        G.add_edge(user_node, recipe_node, weight=rating, edge_type='rating')
        if isinstance(ingredients, str):
            try:
                ing_list = eval(ingredients)
                for ingredient in ing_list:
                    G.add_node(ingredient, node_type='ingredient')
                    G.add_edge(recipe_node, ingredient, edge_type='contains_ingredient')
            except:
                logger.warning(f"not parse data of ingredients: {ingredients}")
        if isinstance(nutrition, str):
            try:
                nut_list = eval(nutrition)
                for j, nutrient_value in enumerate(nut_list):
                    if j < len(nutrients) and nutrient_value > 0:
                        nutrient_name = nutrients[j]
                        G.add_edge(recipe_node, nutrient_name, weight=nutrient_value, edge_type='nutrition_value')
            except:
                logger.warning(f"no data of nutrition: {nutrition}")
    analyze_meta_paths(G)
    return G, health_status

def analyze_meta_paths(G):
    meta_paths = [
        "user->recipe",
        "user->recipe->ingredient",
        "user->recipe->nutrition",
        "recipe->ingredient",
        "recipe->nutrition",
        "user->recipe->ingredient->nutrition"
    ]
    path_counts = {path: 0 for path in meta_paths}
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'user':
            recipes = list(G.neighbors(node))
            path_counts["user->recipe"] += len(recipes)
            for recipe in recipes:
                if G.nodes[recipe]['node_type'] == 'recipe':
                    for neighbor in G.neighbors(recipe):
                        if G.nodes[neighbor]['node_type'] == 'ingredient':
                            path_counts["user->recipe->ingredient"] += 1
                        elif G.nodes[neighbor]['node_type'] == 'nutrition':
                            path_counts["user->recipe->nutrition"] += 1
                    ingredients = [n for n in G.neighbors(recipe) if G.nodes[n]['node_type'] == 'ingredient']
                    nutritions = [n for n in G.neighbors(recipe) if G.nodes[n]['node_type'] == 'nutrition']
                    path_counts["user->recipe->ingredient->nutrition"] += len(ingredients) * len(nutritions)
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'recipe':
            ingredients = [n for n in G.neighbors(node) if G.nodes[n]['node_type'] == 'ingredient']
            path_counts["recipe->ingredient"] += len(ingredients)
            nutritions = [n for n in G.neighbors(node) if G.nodes[n]['node_type'] == 'nutrition']
            path_counts["recipe->nutrition"] += len(nutritions)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cos_sim(anchor, positive) / self.temperature
        pos_sim = torch.exp(pos_sim)
        neg_sim = self.cos_sim(anchor.unsqueeze(1), negative) / self.temperature
        neg_sim = torch.exp(neg_sim).sum(dim=1)
        loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        return loss.mean()

class NutritionAwareEmbedding(nn.Module):
    def __init__(self, num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, health_status):
        super(NutritionAwareEmbedding, self).__init__()
        self.health_status = health_status
        self.healthy_indices = []
        self.unhealthy_indices = []
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.nutrition_embedding = nn.Embedding(num_nutrition, embedding_dim)
        self.health_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.contrastive_loss = ContrastiveLoss(temperature=0.7)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, user_idx, recipe_idx, ingredient_idx, nutrition_idx):
        user_emb = self.user_embedding(user_idx)
        recipe_emb = self.recipe_embedding(recipe_idx)
        ingredient_emb = self.ingredient_embedding(ingredient_idx)
        nutrition_emb = self.nutrition_embedding(nutrition_idx)
        concat_emb = torch.cat([user_emb, recipe_emb, ingredient_emb, nutrition_emb], dim=-1)
        health_emb = self.health_projection(concat_emb)
        return health_emb

    def sample_contrastive_examples(self, recipe_indices):
        batch_size = recipe_indices.size(0)
        positive_samples = []
        negative_samples = []
        for i in range(batch_size):
            ridx = recipe_indices[i].item()
            is_healthy = self.health_status.get(ridx, False)
            if is_healthy:
                if len(self.healthy_indices) > 1:
                    pos_idx = random.choice([idx for idx in self.healthy_indices if idx != ridx])
                else:
                    pos_idx = ridx
                negative_pool = self.unhealthy_indices
            else:
                if len(self.unhealthy_indices) > 1:
                    pos_idx = random.choice([idx for idx in self.unhealthy_indices if idx != ridx])
                else:
                    pos_idx = ridx
                negative_pool = self.healthy_indices
            if negative_pool:
                neg_idx = random.choice(negative_pool)
            else:
                neg_idx = ridx
            with torch.no_grad():
                neg_emb = self.recipe_embedding(torch.tensor([neg_idx], device=recipe_indices.device))
            positive_samples.append(pos_idx)
            negative_samples.append(neg_emb)
        with torch.no_grad():
            positive_embs = self.recipe_embedding(torch.tensor(positive_samples, device=recipe_indices.device))
        negative_embs = torch.cat(negative_samples, dim=0)
        return positive_embs, negative_embs

    def compute_loss(self, user_idx, recipe_idx, ingredient_idx, nutrition_idx, rating):
        health_emb = self.forward(user_idx, recipe_idx, ingredient_idx, nutrition_idx)
        rating_pred = health_emb[:, 0]
        rating_loss = F.mse_loss(rating_pred, rating.float())
        positive_embs, negative_embs = self.sample_contrastive_examples(recipe_idx)
        recipe_emb = self.recipe_embedding(recipe_idx)
        contrastive_loss = self.contrastive_loss(recipe_emb, positive_embs, negative_embs)
        total_loss = rating_loss + contrastive_loss
        return total_loss, rating_loss, contrastive_loss

class FoodRecommendationDataset(Dataset):
    def __init__(self, df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder):
        self.user_encoder = user_encoder
        self.recipe_encoder = recipe_encoder
        self.ingredient_encoder = ingredient_encoder
        self.nutrition_encoder = nutrition_encoder
        self.user_indices = self.user_encoder.transform(df['user_id'])
        self.recipe_indices = self.recipe_encoder.transform(df['recipe_id'])
        self.ingredient_indices = self.ingredient_encoder.transform(df['ingredients'])
        self.nutrition_indices = self.nutrition_encoder.transform(df['nutrition'])
        self.ratings = df['rating'].values
        self.health_scores = df['health_score'].values

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        return (
            self.user_indices[idx],
            self.recipe_indices[idx],
            self.ingredient_indices[idx],
            self.nutrition_indices[idx],
            self.ratings[idx],
            self.health_scores[idx]
        )

def train_model(df, graph, health_status):
    user_encoder = LabelEncoder()
    recipe_encoder = LabelEncoder()
    ingredient_encoder = LabelEncoder()
    nutrition_encoder = LabelEncoder()
    user_encoder.fit(df['user_id'].unique())
    recipe_encoder.fit(df['recipe_id'].unique())
    ingredient_encoder.fit(df['ingredients'].apply(lambda x: str(eval(x)) if isinstance(x, str) else str(x)).unique())
    nutrition_encoder.fit(df['nutrition'].apply(lambda x: str(eval(x)) if isinstance(x, str) else str(x)).unique())
    num_users = len(user_encoder.classes_)
    num_recipes = len(recipe_encoder.classes_)
    num_ingredients = len(ingredient_encoder.classes_)
    num_nutrition = len(nutrition_encoder.classes_)
    dataset = FoodRecommendationDataset(df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    embedding_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NutritionAwareEmbedding(
        num_users, num_recipes, num_ingredients, num_nutrition,
        embedding_dim, health_status
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        rating_loss = 0.0
        contrastive_loss = 0.0
        for batch in dataloader:
            user_idx, recipe_idx, ingredient_idx, nutrition_idx, rating_val, health_score = batch
            user_idx = user_idx.to(device)
            recipe_idx = recipe_idx.to(device)
            ingredient_idx = ingredient_idx.to(device)
            nutrition_idx = nutrition_idx.to(device)
            rating_val = rating_val.to(device)
            optimizer.zero_grad()
            loss, r_loss, c_loss = model.compute_loss(user_idx, recipe_idx, ingredient_idx, nutrition_idx, rating_val)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            rating_loss += r_loss.item()
            contrastive_loss += c_loss.item()
        avg_total_loss = total_loss / len(dataloader)
        avg_rating_loss = rating_loss / len(dataloader)
        avg_contrastive_loss = contrastive_loss / len(dataloader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - "
                    f"Total Loss: {avg_total_loss:.4f}, "
                    f"Nla Loss: {avg_rating_loss:.4f}, "
                    f"Contrastive Loss: {avg_contrastive_loss:.4f}")
    return model, user_encoder, recipe_encoder

def evaluate_recommendations(model, df, user_encoder, recipe_encoder):
    all_recipe_indices = torch.arange(len(recipe_encoder.classes_))
    recipe_embeddings = model.recipe_embedding(all_recipe_indices)
    recipe_embeddings = F.normalize(recipe_embeddings, p=2, dim=1)
    user_history = df.groupby('user_id')['recipe_id'].apply(set).to_dict()
    precision_at_k = []
    recall_at_k = []
    for user_id, rated_recipes in user_history.items():
        user_idx = user_encoder.transform([user_id])[0]
        user_idx_tensor = torch.tensor([user_idx])
        user_emb = model.user_embedding(user_idx_tensor)
        similarity = torch.matmul(user_emb, recipe_embeddings.T).squeeze()
        user_rated = [recipe_encoder.transform([rid])[0] for rid in rated_recipes]
        similarity[user_rated] = -10
        k = 10
        _, top_indices = torch.topk(similarity, k)
        recommended_recipes = [recipe_encoder.inverse_transform([idx])[0] for idx in top_indices]
        hits = [1 for rid in recommended_recipes if rid in rated_recipes]
        precision = len(hits) / k if k > 0 else 0
        recall = len(hits) / len(rated_recipes) if len(rated_recipes) > 0 else 0
        precision_at_k.append(precision)
        recall_at_k.append(recall)
    avg_precision = np.mean(precision_at_k)
    avg_recall = np.mean(recall_at_k)
    logger.info(f"avg_precision@{k}: {avg_precision:.4f}")
    logger.info(f"avg_recall@{k}: {avg_recall:.4f}")
    return avg_precision, avg_recall

def main():
    df = load_and_preprocess_data("Food_Dataset.zip")
    graph, health_status = build_food_graph(df)
    model, user_encoder, recipe_encoder = train_model(df, graph, health_status)
    evaluate_recommendations(model, df, user_encoder, recipe_encoder)