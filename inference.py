import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from peft import PeftModel
from transformers import BlipProcessor, BlipForQuestionAnswering

# Clean up the answer to only return the first word
def clean_answer(answer):
    return re.findall(r"\b\w+\b", answer.strip().lower())[0] if re.findall(r"\b\w+\b", answer.strip().lower()) else ""

# Generate an answer using BLIP
def predict_blip(image_path, question, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            num_beams=1
        )
    decoded = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_answer(decoded)

# Generate answers for all rows in dataset
def generate_answers(df, image_folder, processor, model, device):
    answers = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
        image_path = os.path.join(image_folder, row["image_name"])
        question = row["question"]
        pred = predict_blip(image_path, question, processor, model, device)
        answers.append(pred)
    return answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()
    print("Arguments:", args)
    # Load CSV
    df = pd.read_csv(args.csv_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load base BLIP model
    base_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        device_map="auto",
        # load_in_8bit=True,
        torch_dtype=torch.float16
    )

    # Load processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    # Load LoRA fine-tuned weights
    lora_path = "sohith18/blip-lora-vqa"
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # Generate answers
    generated_answers = generate_answers(df, args.image_dir, processor, model, device)

    # Save results
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
