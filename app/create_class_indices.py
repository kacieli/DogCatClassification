import json
import os

# Define dataset path
base_path = "dog_cat_cleaned"

# Initialize mapping
breed_to_index = {}
index = 0

# Iterate through folders
for animal_type in ["cat", "dog"]:
    folder_path = os.path.join(base_path, animal_type)
    if os.path.exists(folder_path):
        # Sort folders case-insensitively
        breed_folders = sorted(os.listdir(folder_path), key=lambda x: x.lower())
        for breed in breed_folders:
            breed_to_index[breed] = index
            index += 1

# Save to a JSON file
with open("breed_to_index.json", "w") as json_file:
    json.dump(breed_to_index, json_file, indent=4)

# print("Mapping saved to 'breed_to_index.json'")
