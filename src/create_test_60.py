import os
from datasets import load_dataset, load_from_disk, Dataset
import random

SEED = 42
random.seed(SEED)

dataset = load_dataset("codeparrot/apps", split="test")

# Group data by difficulty
difficulty_groups = {
    "interview": [],
    "competition": [],
    "introductory": []
}

for record in dataset:
    if record["difficulty"] in difficulty_groups:
        difficulty_groups[record["difficulty"]].append(record)

# Randomly sample from each difficulty level
samples = {
    "interview": random.sample(difficulty_groups["interview"], min(20, len(difficulty_groups["interview"]))),
    "competition": random.sample(difficulty_groups["competition"], min(20, len(difficulty_groups["competition"]))),
    "introductory": random.sample(difficulty_groups["introductory"], min(20, len(difficulty_groups["introductory"])))
}

save_path = "../data/test_60/"
os.makedirs(save_path, exist_ok=True)

for difficulty, subset in samples.items():
  subset_dataset = Dataset.from_list(subset)  # Convert the list of samples into a Dataset object
  subset_dataset.save_to_disk(save_path+difficulty)  # Save the dataset to disk
  print(f"Saved {len(subset)} records for '{difficulty}' difficulty to {difficulty}_subset/")