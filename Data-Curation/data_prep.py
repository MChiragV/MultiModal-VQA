""" 
First setup Vertex API account.
Run these commands to login:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform
"""

from google import genai
from google.genai.types import GenerateContentConfig, Part
import os
import pandas as pd
import json
from utils import *
import csv
import random

random.seed(42)

df = pd.read_csv(IMAGES_METADATA_CSV)

json_files = os.listdir(LISTINGS_METADATA_PATH)

json_file_paths = [os.path.join(LISTINGS_METADATA_PATH, filename) for filename in json_files]

images_metadata = []
for filepath in json_file_paths:
    print(f"[INFO] Reading {filepath}")
    with open(filepath, mode="r", encoding="utf-8") as read_file:
        json_strings = read_file.readlines()
        for json_string in json_strings:
            images_metadata.append(json.loads(json_string))

random.shuffle(images_metadata)

# Add your project ID and location
client = genai.Client(vertexai=True, project="PROJECT_ID", location="LOCATION")


MODEL_ID = (
    "projects/PROJECT_ID"
    "/locations/LOCATION"
    "/publishers/google"
    "/models/gemini-2.0-flash-001"
) # Add your project ID and location

PROMPT = "I am preparing a dataset to train a Visual Question Answering (VQA) model. \
    I have a set of images and corresponding metadata from Amazon product listings (Amazon Berkeley Object Dataset). \
    Using the image generate questions with single unambiguous one-word answers in English which should be answerable by \"SOLELY\" looking at the image without providing any other data.\
    Use its metadata to verify the answers. \
    Each question has to be independent of the others.\
    Since I need to automate parsing these question and answers, please provide them in CSV format: question, answer.\
    Please do not generate anything else other than question and answers as it makes it difficult to write an automated parser."



def call_with_inline(client, image_path, mime_type, metadata):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    image_part = Part.from_bytes(data=img_bytes, mime_type=mime_type)  # inline :contentReference[oaicite:6]{index=6}
    gen_config = gen_config = GenerateContentConfig(
        temperature=0.0,      # greedy decoding
        top_k=1,              # pick single best token
        top_p=1.0,            # include all tokens
        seed=42,              # lock in random state
        max_output_tokens=256 
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        config = gen_config,
        contents=[PROMPT, metadata, image_part]
    )
    return response.text

if __name__ == "__main__":
    img_count = 0
    seen_paths = set()
    with open("main_image_qa_8.csv", "w") as write_file:
        for i, img_metadata in enumerate(images_metadata):
            if (i+1) % 100 == 0:
                print(f"INFO: Images Processed: {(i+1)}/{len(images_metadata)} Images Used: {img_count}/{len(images_metadata)}")

            try:
                req_info = get_required_info(img_metadata)
                if len(req_info) < 5:
                    continue

                metadata = "\n".join(req_info)
                image_path = get_main_image_path(img_metadata, df)

                if image_path in seen_paths:
                    continue
                else:
                    seen_paths.add(image_path)
                        
                mime_type = "image/jpeg" if image_path.endswith("jpg") else "image/png"
                response = call_with_inline(client, image_path, mime_type, metadata)

                lis = response.split("\n")
                qas = list(csv.reader(lis))
                if not qas:
                    continue

                if qas[0] == ['```csv']:
                    del qas[0]
                    del qas[-1]
                del qas[0]

                if qas:
                    for qa in qas:
                        if not qa:
                            continue
                        write_file.write(f"{image_path}, \"{qa[0]}\", \"{qa[1]}\"\n")
                img_count += 1
            except Exception as e:
                print("Error in inline approach:", e)