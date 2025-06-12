import os
import kagglehub

try:
    # === Set custom directory for dataset download ===
    custom_dir = "/home/soufianejd/datasets/violence"
    os.environ["KAGGLEHUB_DIR"] = custom_dir

    # === Download the dataset ===
    path = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")

    print("Dataset downloaded successfully.")
    print("Path to dataset files:", path)

except Exception as e:
    print("❌ Error downloading dataset:", str(e))
