# 🛰️ Satellite Land Cover Classifier (EuroSAT + Vision Transformer)

This repository contains an end-to-end pipeline for training a state-of-the-art **Vision Transformer (ViT)** model on satellite images to perform **Land Use and Land Cover (LULC) Classification**. 

The model is fine-tuned on the **EuroSAT RGB** dataset (derived from Sentinel-2 satellite imagery) containing **10 land-cover classes** and **27,000 images**. 

We also provide a fully-functional **Gradio web application** designed to be hosted on **Hugging Face Spaces** allowing users to upload satellite imagery and view predictions in real-time.

---

## 📂 Repository Structure

*   `train.py`: A modular and fully parameterized training script. It handles dataset splitting (80/10/10), data augmentations, training via Hugging Face `Trainer`, evaluation, and automatic uploads to the Hugging Face Hub.
*   `app.py`: The Gradio web interface code. It supports loading the default fine-tuned model or loading any custom-trained model dynamically from the Hugging Face Hub.
*   `requirements.txt`: Python package requirements for training and serving.

---

## ⚡ Step-by-Step Training Guide (Google Colab)

Since training a Vision Transformer is computationally intensive, it is highly recommended to run the training on a **GPU** in Google Colab (the free T4 GPU is perfect).

### Step 1: Create a Colab Notebook and Connect to a GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Change the runtime type to **GPU**: Go to `Runtime` -> `Change runtime type` -> select `T4 GPU`.

### Step 2: Upload Files & Install Dependencies
Run the following commands in a Colab code cell to install the required libraries:
```bash
!pip install transformers datasets evaluate accelerate scikit-learn
```

Upload `train.py` directly to the Colab workspace (using the folder icon in the left sidebar).

### Step 3: Run the Training Script
Run the script using `!python`. You can customize hyperparameters via command-line arguments:

```bash
!python train.py \
    --model_name "google/vit-base-patch16-224-in21k" \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-5 \
    --output_dir "./eurosat-vit-model"
```

*Note: With a T4 GPU, 3 epochs of training will complete in approximately 10-15 minutes and will yield >97% test accuracy.*

---

## 🤗 Pushing the Model to Hugging Face Hub

If you want to save your trained weights to the Hugging Face Hub directly from your Colab notebook, do the following:

1. Create a Write access token on [Hugging Face Settings](https://huggingface.co/settings/tokens).
2. Run training with your token and Hub repository ID:

```bash
!python train.py \
    --model_name "google/vit-base-patch16-224-in21k" \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-5 \
    --output_dir "./eurosat-vit-model" \
    --hub_model_id "your-hf-username/vit-base-eurosat" \
    --hf_token "your_hf_write_token"
```

Your model card, weights, and image processor will automatically be uploaded to `https://huggingface.co/your-hf-username/vit-base-eurosat`.

---

## 🚀 Deploying the Web App to Hugging Face Spaces

You can host your interactive classifier web app on Hugging Face Spaces for free:

1. Log in to [Hugging Face](https://huggingface.co/) and click on **New Space** (top right corner).
2. Enter a name (e.g. `eurosat-classifier`).
3. Select **Gradio** as the Space SDK.
4. Keep the repository public and choose the **Blank** template (free CPU basic tier is fine).
5. Once created, upload the following files from this repo to the Space:
   *   `app.py`
   *   `requirements.txt`
6. Edit `app.py` directly on the Space, changing the `DEFAULT_MODEL` variable value to your custom Hugging Face model repository ID:
   ```python
   DEFAULT_MODEL = "your-hf-username/vit-base-eurosat"
   ```
7. Hugging Face will automatically build and host your app! 

---

## 📊 EuroSAT Classes Reference

Your model will learn to classify images into these 10 categories:
*   `AnnualCrop` - Cultivated farm fields.
*   `Forest` - Dense trees and woodland.
*   `HerbaceousVegetation` - Wild meadows and grasslands.
*   `Highway` - Major roads and freeways.
*   `Industrial` - Commercial areas, factories, and warehouses.
*   `Pasture` - Fields for livestock grazing.
*   `PermanentCrop` - Vineyards, orchards, and perennial farms.
*   `Residential` - Housing complexes and suburbs.
*   `River` - Waterways and streams.
*   `SeaLake` - Lakes, oceans, and reservoirs.
