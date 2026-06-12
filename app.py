import os
import gradio as gr
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# Default fallback model fine-tuned on EuroSAT RGB
DEFAULT_MODEL = "aedriansagap/vit-base-eurosat"

# Cache loaded models to prevent reloading on every inference
MODEL_CACHE = {}

def get_model_and_processor(model_id):
    """Loads and caches the model and processor from Hugging Face Hub."""
    if model_id not in MODEL_CACHE:
        print(f"Loading model and processor for {model_id}...")
        try:
            processor = ViTImageProcessor.from_pretrained(model_id)
            model = ViTForImageClassification.from_pretrained(model_id)
            MODEL_CACHE[model_id] = (model, processor)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}': {str(e)}")
    return MODEL_CACHE[model_id]

def predict(image, custom_model_id):
    """Performs inference on the uploaded image using the selected model."""
    if not image:
        return "Please upload an image.", {}
    
    # Determine which model to use
    model_id = custom_model_id.strip() if custom_model_id and custom_model_id.strip() else DEFAULT_MODEL
    
    try:
        model, processor = get_model_and_processor(model_id)
    except Exception as e:
        return f"Error loading model: {str(e)}", {}
        
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Run forward pass (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Compute probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Map probabilities to classes
    id2label = model.config.id2label
    results = {}
    for idx, prob in enumerate(probs):
        class_name = id2label.get(idx, id2label.get(str(idx), f"Class {idx}"))
        results[class_name] = float(prob)
        
    # Sort results to get the top prediction for the text label
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_label = sorted_results[0][0]
    top_conf = sorted_results[0][1]
    
    summary_text = f"### Primary Classification: **{top_label}** (Confidence: {top_conf:.1%})"
    
    return summary_text, results

# Custom CSS for modern, premium appearance (Glassmorphism & Clean Typography)
custom_css = """
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.gradio-container {
    background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%) !important;
    color: #f3f4f6 !important;
}
.header-box {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.header-box h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.footer-text {
    text-align: center;
    margin-top: 2rem;
    color: #6b7280;
    font-size: 0.875rem;
}
"""

# Build Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome(primary_hue="blue", secondary_hue="indigo")) as demo:
    with gr.Column(elem_classes="header-box"):
        gr.Markdown(
            """
            # 🛰️ EuroSAT Satellite Land Cover Classifier
            ### Harnessing Vision Transformers (ViT) to Analyze Earth from Space
            This application uses a pre-trained **Vision Transformer (ViT)** model fine-tuned on the **EuroSAT RGB** dataset.
            The model analyzes Sentinel-2 satellite images (64x64 pixels) and classifies them into one of 10 land cover categories.
            """
        )
        
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Panel")
            image_input = gr.Image(type="pil", label="Upload Satellite Image")
            
            with gr.Accordion("⚙️ Advanced Options", open=False):
                model_input = gr.Textbox(
                    label="Custom Model ID (Hugging Face Hub)",
                    placeholder=f"e.g., your-username/vit-base-eurosat (Leave blank to use default)",
                    value=""
                )
                
            submit_btn = gr.Button("Analyze Land Cover", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Classification Results")
            summary_output = gr.Markdown("Run classification to see primary class.")
            label_output = gr.Label(num_top_classes=5, label="Class Probabilities")

    # Wire up the execution trigger
    submit_btn.click(
        fn=predict,
        inputs=[image_input, model_input],
        outputs=[summary_output, label_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 📖 EuroSAT Class Guide:
        *   **AnnualCrop**: Regularly cultivated farm fields.
        *   **Forest**: Dense woodland/canopy cover.
        *   **HerbaceousVegetation**: Grasslands, wild shrubs, and meadows.
        *   **Highway**: Major roads, freeways, and motorways.
        *   **Industrial**: Factory complexes, warehouses, and commercial zones.
        *   **Pasture**: Fenced fields used for livestock grazing.
        *   **PermanentCrop**: Orchards, vineyards, and fruit farms.
        *   **Residential**: Houses, apartments, and neighborhoods.
        *   **River**: Moving streams and channels of water.
        *   **SeaLake**: Estuaries, oceans, reservoirs, and lakes.
        """,
        elem_classes="footer-text"
    )

if __name__ == "__main__":
    demo.launch()
