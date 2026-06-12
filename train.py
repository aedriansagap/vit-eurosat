#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import evaluate
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter
)
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Vision Transformer (ViT) on EuroSAT")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Pretrained model name or path from Hugging Face hub"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="giswqs/EuroSAT_RGB",
        help="EuroSAT dataset name on Hugging Face hub"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eurosat-vit-outputs",
        help="Output directory where model predictions and checkpoints will be written"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face hub repository id (e.g. username/vit-base-eurosat)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token with write permissions"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("--- 1. Parsing Arguments & Environment Check ---")
    print(f"Using model: {args.model_name}")
    print(f"Using dataset: {args.dataset_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Log in to Hugging Face if token provided or hub push requested
    if args.hf_token:
        print("Logging in to Hugging Face...")
        login(token=args.hf_token)
    elif args.hub_model_id and "HF_TOKEN" not in os.environ:
        print("Warning: Pushing to hub requested but hf_token not supplied and HF_TOKEN env var not found.")

    print("\n--- 2. Loading Dataset ---")
    # Load EuroSAT
    raw_dataset = load_dataset(args.dataset_name)
    
    # giswqs/EuroSAT_RGB only contains a 'train' split. Split into 80% train, 10% validation, 10% test.
    # First split train and test (90% / 10%)
    split_dataset = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
    # Then split the 20% test into 10% validation and 10% test
    val_test_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    
    dataset = {
        "train": split_dataset["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    }
    
    print(f"Dataset splits: Train={len(dataset['train'])}, Val={len(dataset['validation'])}, Test={len(dataset['test'])}")
    
    # Retrieve label mappings
    labels = dataset["train"].features["label"].names
    num_labels = len(labels)
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    print(f"Labels found: {labels}")

    print("\n--- 3. Preparing Transforms & Preprocessing ---")
    image_processor = ViTImageProcessor.from_pretrained(args.model_name)
    
    # Get image resolution expected by model (typically 224)
    size = image_processor.size["height"]
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    
    # Define augmentations for training
    train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ToTensor(),
        normalize,
    ])
    
    # Simple validation/testing transform (no augmentation)
    val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])
    
    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [
            train_transforms(img.convert("RGB")) for img in example_batch["image"]
        ]
        return example_batch
        
    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [
            val_transforms(img.convert("RGB")) for img in example_batch["image"]
        ]
        return example_batch

    # Set the transforms dynamically (saves memory)
    dataset["train"].set_transform(preprocess_train)
    dataset["validation"].set_transform(preprocess_val)
    dataset["test"].set_transform(preprocess_val)

    # Data collator to batch tensors
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch])
        }

    print("\n--- 4. Initializing Model ---")
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # Ignore mismatch on classification head weights
    )

    print("\n--- 5. Configuring Metrics & Training Arguments ---")
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        remove_unused_columns=False, # Necessary when using dynamic set_transform
        push_to_hub=True if args.hub_model_id else False,
        hub_model_id=args.hub_model_id,
        hub_strategy="every_save",
        report_to="none" # Disable logging to WandB/Tensorboard for simplicity
    )

    print("\n--- 6. Running Training Loop ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=image_processor
    )
    
    trainer.train()

    print("\n--- 7. Evaluating Model on Test Set ---")
    eval_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    print(f"Test Set Metrics: {eval_metrics}")

    print("\n--- 8. Saving Model Locally ---")
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print(f"Model and processor saved successfully to '{args.output_dir}'")

    if args.hub_model_id:
        print("\n--- 9. Pushing to Hugging Face Hub ---")
        trainer.push_to_hub(commit_message="Training finished successfully!")
        image_processor.push_to_hub(args.hub_model_id)
        print(f"Model pushed to HF Hub: https://huggingface.co/{args.hub_model_id}")

if __name__ == "__main__":
    main()
