"""
hosting.py – Push deployment files to Hugging Face Spaces.
Run this script after updating the HF_TOKEN and HF_SPACE_REPO variables.
"""
import os
from huggingface_hub import HfApi, login

HF_TOKEN     = os.environ.get("HF_TOKEN", "<YOUR_HF_TOKEN>")
HF_SPACE_REPO = "DineshDSV/tourism-wellness-app"  # ← update

login(token=HF_TOKEN)
api = HfApi()

# Create the Space (Streamlit SDK)
api.create_repo(
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    space_sdk="docker", # Changed from "streamlit" to "docker"
    exist_ok=True
)
print(f"Space ready: https://huggingface.co/spaces/{HF_SPACE_REPO}")

# Upload deployment files
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
for fname in ["app.py", "requirements.txt", "Dockerfile"]:
    fpath = os.path.join(DEPLOY_DIR, fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=HF_SPACE_REPO,
            repo_type="space",
            token=HF_TOKEN
        )
        print(f"Uploaded: {fname}")

print("\nDeployment complete! App should be live at:")
print(f"https://huggingface.co/spaces/{HF_SPACE_REPO}")
