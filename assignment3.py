import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Load the ratings data
ratings = pd.read_csv("data/ratings.csv")

# Check the dataset
print(ratings.head())
print(ratings.info())

# Random split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

# Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Sort by timestamp
ratings = ratings.sort_values(by='timestamp')

# Temporal split (e.g., use 80% of the timeline for training)
split_time = ratings['timestamp'].quantile(0.8)
train_data_temp = ratings[ratings['timestamp'] <= split_time]
test_data_temp = ratings[ratings['timestamp'] > split_time]

print("Temporal training data shape:", train_data_temp.shape)
print("Temporal testing data shape:", test_data_temp.shape)

# Define a reader
reader = Reader(rating_scale=(0.5, 5.0))

# Load datasets into Surprise format
data_random = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)
data_temp = Dataset.load_from_df(train_data_temp[['userId', 'movieId', 'rating']], reader)

# Train on random split
trainset_random = data_random.build_full_trainset()
model_random = SVD()
model_random.fit(trainset_random)

# Train on temporal split
trainset_temp = data_temp.build_full_trainset()
model_temp = SVD()
model_temp.fit(trainset_temp)

# Convert test data to Surprise format
testset_random = list(zip(test_data['userId'], test_data['movieId'], test_data['rating']))

# Predict and evaluate
predictions_random = model_random.test(testset_random)
rmse_random = accuracy.rmse(predictions_random)
print(f"RMSE (Random Split): {rmse_random:.4f}")

# Convert test data to Surprise format
testset_temp = list(zip(test_data_temp['userId'], test_data_temp['movieId'], test_data_temp['rating']))

# Predict and evaluate
predictions_temp = model_temp.test(testset_temp)
rmse_temp = accuracy.rmse(predictions_temp)
print(f"RMSE (Temporal Split): {rmse_temp:.4f}")

# Predict a rating for user 1 and movie 10
user_id = 1
movie_id = 10
predicted_rating = model_random.predict(user_id, movie_id).est
print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating:.2f}")

