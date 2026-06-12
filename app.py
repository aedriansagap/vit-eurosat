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

# Custom visual styles and environmental descriptions for EuroSAT classes
INFO_MAP = {
    "AnnualCrop": {
        "icon": "🌾",
        "color": "#eab308", # Amber/Yellow
        "desc": "Regularly cultivated farm fields (e.g. wheat, soy, vegetables). Essential for global food supply, but intensive farming can lead to soil depletion and chemical runoff."
    },
    "Forest": {
        "icon": "🌲",
        "color": "#22c55e", # Green
        "desc": "Dense woodlands, deciduous or coniferous forests. Vital for carbon sequestration, producing oxygen, protecting soil quality, and supporting rich biodiversity."
    },
    "HerbaceousVegetation": {
        "icon": "🌱",
        "color": "#84cc16", # Lime Green
        "desc": "Wild grasslands, savannahs, meadows, and scrublands. Crucial habitats for local fauna, insect pollinators, and wild herbaceous plants."
    },
    "Highway": {
        "icon": "🛣️",
        "color": "#9ca3af", # Gray
        "desc": "Paved roads, major highways, and transportation infrastructure. Connects regions and urban hubs, but fragments ecosystems and increases local emissions."
    },
    "Industrial": {
        "icon": "🏭",
        "color": "#ef4444", # Red
        "desc": "Industrial parks, factory complexes, warehouses, and commercial centers. Hotspots of manufacturing and distribution, requiring active pollution monitoring."
    },
    "Pasture": {
        "icon": "🐄",
        "color": "#10b981", # Emerald
        "desc": "Grassy fields used specifically for livestock grazing and animal agriculture. Requires managed rotational grazing to prevent soil erosion."
    },
    "PermanentCrop": {
        "icon": "🍇",
        "color": "#f97316", # Orange
        "desc": "Perennial cropland like orchards, vineyards, and olive groves. Unlike annual crops, these plants stay in the soil for multiple seasons."
    },
    "Residential": {
        "icon": "🏡",
        "color": "#3b82f6", # Blue
        "desc": "Urban housing, suburbs, apartment blocks, and domestic developments. Center of human inhabitation, needing sustainable green spaces and stormwater drainage."
    },
    "River": {
        "icon": "🌊",
        "color": "#06b6d4", # Cyan
        "desc": "Flowing freshwater bodies, streams, and narrow waterways. Crucial corridors for regional ecology, drinking water supply, and transport."
    },
    "SeaLake": {
        "icon": "⛵",
        "color": "#2563eb", # Indigo/Blue
        "desc": "Estuaries, deep lakes, seas, and oceans. Covers the majority of Earth's surface, regulating global temperature and hosting complex marine systems."
    }
}

def predict(image, custom_model_id):
    """Performs inference on the uploaded image using the selected model."""
    if not image:
        return "<p style='color: #ef4444; font-weight: 600; font-family: sans-serif;'>Please upload an image.</p>", {}
    
    # Determine which model to use
    model_id = custom_model_id.strip() if custom_model_id and custom_model_id.strip() else DEFAULT_MODEL
    
    try:
        model, processor = get_model_and_processor(model_id)
    except Exception as e:
        return f"<p style='color: #ef4444; font-weight: 600; font-family: sans-serif;'>Error loading model: {str(e)}</p>", {}
        
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
    
    # Build a premium HTML result badge and card
    info = INFO_MAP.get(top_label, {
        "icon": "🛰️",
        "color": "#60a5fa",
        "desc": "Custom classified land cover category."
    })
    
    icon = info["icon"]
    color = info["color"]
    desc = info["desc"]
    
    warning_html = ""
    if top_conf < 0.60:
        warning_html = f"""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; color: #f59e0b;">
            <span style="font-size: 1.1rem; line-height: 1;">⚠️</span>
            <span style="font-size: 0.85rem; font-weight: 600; font-family: sans-serif;">Low Confidence Prediction: Please verify the image quality.</span>
        </div>
        """
        
    summary_html = f"""
    {warning_html}
    <div style="background: rgba(255, 255, 255, 0.03); border-left: 6px solid {color}; padding: 1.25rem; border-radius: 8px; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px);">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; flex-wrap: wrap;">
            <span style="font-size: 1.75rem; line-height: 1;">{icon}</span>
            <h3 style="margin: 0; font-size: 1.35rem; color: {color}; font-weight: 700; font-family: sans-serif; letter-spacing: 0.5px;">{top_label}</h3>
            <span style="background: {color}22; color: {color}; padding: 0.25rem 0.6rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; margin-left: auto; font-family: sans-serif;">
                {top_conf:.1%} Confidence
            </span>
        </div>
        <p style="margin: 0; line-height: 1.5; color: #d1d5db; font-size: 0.9rem; font-family: sans-serif;">{desc}</p>
    </div>
    """
    
    return summary_html, results

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
            summary_output = gr.HTML(value="<div style='color: #6b7280; font-size: 0.9rem; font-family: sans-serif;'>Upload an image and run classification to see detailed analysis.</div>")
            label_output = gr.Label(num_top_classes=5, label="Class Probabilities")

    # Local examples for quick testing
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    example_images = [
        "Forest.png", "SeaLake.png", "Residential.png", 
        "Industrial.png", "Highway.png", "AnnualCrop.png"
    ]
    example_list = [[os.path.join(examples_dir, filename), ""] for filename in example_images]
    
    gr.Examples(
        examples=example_list,
        inputs=[image_input, model_input],
        outputs=[summary_output, label_output],
        fn=predict,
        cache_examples=False,
        label="💡 Try with these Sample Images"
    )

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
