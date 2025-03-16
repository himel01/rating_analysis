import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from sklearn.metrics import precision_score, recall_score, f1_score, root_mean_squared_error

# Load the ratings data
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Random split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Temporal split
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings.sort_values(by='timestamp')
split_time = ratings['timestamp'].quantile(0.8)
train_data_temp = ratings[ratings['timestamp'] <= split_time]
test_data_temp = ratings[ratings['timestamp'] > split_time]

# Convert ratings to binary
threshold = 4.0
train_data['binary_rating'] = (train_data['rating'] >= threshold).astype(int)
test_data['binary_rating'] = (test_data['rating'] >= threshold).astype(int)
train_data_temp['binary_rating'] = (train_data_temp['rating'] >= threshold).astype(int)
test_data_temp['binary_rating'] = (test_data_temp['rating'] >= threshold).astype(int)

# Prepare data for Surprise
reader = Reader(rating_scale=(0, 1))  # Binary ratings
data_random = Dataset.load_from_df(train_data[['userId', 'movieId', 'binary_rating']], reader)
data_temp = Dataset.load_from_df(train_data_temp[['userId', 'movieId', 'binary_rating']], reader)

# Train on random split
trainset_random = data_random.build_full_trainset()
model_random = SVD()
model_random.fit(trainset_random)

# Train on temporal split
trainset_temp = data_temp.build_full_trainset()
model_temp = SVD()
model_temp.fit(trainset_temp)

# Predict on random split
testset_random = list(zip(test_data['userId'], test_data['movieId'], test_data['binary_rating']))
predictions_random = model_random.test(testset_random)
predicted_binary_random = [1 if pred.est >= 0.5 else 0 for pred in predictions_random]
true_binary_random = test_data['binary_rating'].values

# Predict on temporal split
testset_temp = list(zip(test_data_temp['userId'], test_data_temp['movieId'], test_data_temp['binary_rating']))
predictions_temp = model_temp.test(testset_temp)
predicted_binary_temp = [1 if pred.est >= 0.5 else 0 for pred in predictions_temp]
true_binary_temp = test_data_temp['binary_rating'].values

# Compute metrics for random split
precision_random = precision_score(true_binary_random, predicted_binary_random)
recall_random = recall_score(true_binary_random, predicted_binary_random)
f1_random = f1_score(true_binary_random, predicted_binary_random)

print("Random Split Metrics:")
print(f"Precision: {precision_random:.4f}")
print(f"Recall: {recall_random:.4f}")
print(f"F1-Score: {f1_random:.4f}")

# Compute metrics for temporal split
precision_temp = precision_score(true_binary_temp, predicted_binary_temp)
recall_temp = recall_score(true_binary_temp, predicted_binary_temp)
f1_temp = f1_score(true_binary_temp, predicted_binary_temp)

print("Temporal Split Metrics:")
print(f"Precision: {precision_temp:.4f}")
print(f"Recall: {recall_temp:.4f}")
print(f"F1-Score: {f1_temp:.4f}")