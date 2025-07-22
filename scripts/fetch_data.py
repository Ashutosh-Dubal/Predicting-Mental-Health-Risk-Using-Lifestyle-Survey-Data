import kagglehub
import shutil
import os

# Define target directory (where you want files copied)
target_path = "/Users/ashutoshdubal/PythonProjects/Predicting Mental Health Risk Using Lifestyle Survey Data/data/raw"

# Check if target already has files (you can adjust to check for specific file if needed)
if os.path.exists(target_path) and os.listdir(target_path):
    print("Dataset already exists in target location. Skipping download and copy.")
else:
    print("Dataset not found locally. Downloading...")

    # Download using kagglehub (saves to default cache)
    source_path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")

    # Make sure target folder exists
    os.makedirs(target_path, exist_ok=True)

    # Copy files from source to target
    for file_name in os.listdir(source_path):
        src_file = os.path.join(source_path, file_name)
        dst_file = os.path.join(target_path, file_name)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

    print("Files copied to:", target_path)