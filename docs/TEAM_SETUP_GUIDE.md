# Team Setup Guide - HuggingFace Models

## Problem
The fine-tuned models are **6.8GB** - too large for standard GitHub (100MB file limit).

## Solutions for Your Team

### ✅ Option 1: Train Models Locally (Recommended for Learning)

**Best for**: Educational projects, small teams, reproducibility

**Steps:**
1. Clone the repository
2. Run the training commands (takes ~30-60 minutes)
3. Models saved locally

```bash
# Team member setup
git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector
git checkout feature/provenance-synthid-detection

# Install dependencies
pip install -r requirements.txt

# Prepare dataset
python manage.py prepare_genimage_dataset

# Fine-tune all 3 models (30-60 minutes)
python manage.py finetune_hf_models --model all --epochs 3 --batch-size 4

# Run server
export MEMORY_CONSTRAINED=true FORCE_CPU=true
python manage.py runserver
```

**Pros:**
- ✅ No large file transfer needed
- ✅ Team learns the training process
- ✅ Reproducible results
- ✅ Each member has their own models

**Cons:**
- ⏱️ Takes 30-60 minutes per setup
- 💻 Requires compute resources

---

### ✅ Option 2: Git LFS (GitHub Large File Storage)

**Best for**: Production teams, sharing exact model weights

**Setup (One-time):**

```bash
# Install Git LFS
# macOS:
brew install git-lfs

# Ubuntu/Debian:
sudo apt-get install git-lfs

# Windows:
# Download from https://git-lfs.github.com/

# Initialize Git LFS
cd AI_image_detector
git lfs install

# Track model files
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.pt"
git lfs track "hf_finetuned_models/**"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add hf_finetuned_models/
git commit -m "Add fine-tuned models with Git LFS"
git push -u origin feature/provenance-synthid-detection
```

**Team members clone:**
```bash
git clone https://github.com/DMKALALA/AI_image_detector.git
# Git LFS automatically downloads large files
```

**Pros:**
- ✅ Exact same models for all team members
- ✅ Integrated with GitHub
- ✅ Version controlled

**Cons:**
- 💰 GitHub LFS pricing: 1GB free, then $5/month per 50GB
- 📦 Requires Git LFS installation

---

### ✅ Option 3: Cloud Storage Share (Free)

**Best for**: Quick sharing, no Git LFS needed

**Services:**
- Google Drive
- Dropbox  
- OneDrive
- Hugging Face Hub (free model hosting!)

**Using Hugging Face Hub (Recommended):**

```bash
# 1. Upload your fine-tuned models
pip install huggingface_hub
huggingface-cli login

# Upload each model
huggingface-cli upload DMKALALA/vit-ai-detector-finetuned hf_finetuned_models/vit_ai_detector_finetuned/
huggingface-cli upload DMKALALA/ai-human-detector-finetuned hf_finetuned_models/ai_human_detector_finetuned/
huggingface-cli upload DMKALALA/ai-classifier-finetuned hf_finetuned_models/ai_classifier_finetuned/
```

**Update code to use HF Hub models:**

Edit `detector/huggingface_models.py`:
```python
# Instead of local paths, use your HF Hub models
model_name_vit = "DMKALALA/vit-ai-detector-finetuned"  # Your uploaded model
model_name_ai_human = "DMKALALA/ai-human-detector-finetuned"
model_name_classifier = "DMKALALA/ai-classifier-finetuned"
```

**Pros:**
- ✅ Free unlimited model hosting on HuggingFace
- ✅ Automatic download for team
- ✅ No Git LFS needed
- ✅ Models publicly available or private

**Cons:**
- 🔧 Requires HuggingFace account
- 📤 One-time upload needed

---

### ✅ Option 4: Shared Drive Link

**Best for**: Quick team sharing

**Steps:**
1. Compress models: `tar -czf hf_models.tar.gz hf_finetuned_models/`
2. Upload to Google Drive/Dropbox
3. Share link with team
4. Team downloads and extracts in project root

**Add to README:**
```markdown
## Fine-Tuned Models

Download from: [Google Drive Link]
Extract to project root:
  tar -xzf hf_models.tar.gz
```

---

## 📋 Recommended Approach for Your Team

### For Educational/Learning Projects:
**Use Option 1** (Train locally) - Best for learning and reproducibility

### For Production/Collaboration:
**Use Option 3** (HuggingFace Hub) - Free, professional, easy

---

## 🚀 Current Branch Status

Your feature branch has:
- ✅ All code changes
- ✅ Training scripts
- ✅ Documentation
- ❌ Fine-tuned models (excluded from git)

**What to push:**
- Code only (small, ~few MB)
- Models excluded (will be regenerated or downloaded)

---

## 📝 Instructions for Your Team

Add this to your README.md:

```markdown
## Setup with Fine-Tuned Models

### Quick Start (Use Pre-trained)
```bash
git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector
git checkout feature/provenance-synthid-detection
pip install -r requirements.txt
python manage.py runserver
# Models will use pre-trained HuggingFace models (still good!)
```

### Full Setup (Fine-tune locally)
```bash
git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector  
git checkout feature/provenance-synthid-detection
pip install -r requirements.txt

# Train models (30-60 minutes)
python manage.py prepare_genimage_dataset
python manage.py finetune_hf_models --model all --epochs 3

# Run server
python manage.py runserver
# Will automatically use fine-tuned models (100% val accuracy)
```
```

---

## 💡 My Recommendation

**Push the code WITHOUT models**, then use **Option 1** (local training):

**Why?**
1. Small push (just code)
2. Team learns the process
3. Reproducible
4. No extra services needed
5. Models are already achieving 100% validation accuracy - easy to reproduce

Want me to push just the code now (without models)?

