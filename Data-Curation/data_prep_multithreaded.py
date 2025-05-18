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
from google.api_core.exceptions import TooManyRequests, InternalServerError

import os
import pandas as pd
import json
import csv
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *

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

lock = threading.Lock()
seen_paths = set()

def call_with_inline(client, image_path, mime_type, metadata):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    image_part = Part.from_bytes(data=img_bytes, mime_type=mime_type)

    gen_config = GenerateContentConfig(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=42,
        max_output_tokens=256
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        config=gen_config,
        contents=[PROMPT, metadata, image_part]
    )
    return response.text

def safe_call(image_path, mime_type, metadata):
    retries = 3
    for attempt in range(retries):
        try:
            # Add your project ID and location
            client = genai.Client(vertexai=True, project="PROJECT_ID", location="LOCATION")
            return call_with_inline(client, image_path, mime_type, metadata)
        except (TooManyRequests, InternalServerError) as e:
            wait = 2 ** attempt
            print(f"[WARN] API error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Non-retryable error: {e}")
            break
    return None

def process_image(i, img_metadata):
    try:
        req_info = get_required_info(img_metadata)
        if len(req_info) < 5:
            return None

        metadata = "\n".join(req_info)
        image_path = get_main_image_path(img_metadata, df)

        with lock:
            if image_path in seen_paths:
                return None
            seen_paths.add(image_path)

        mime_type = "image/jpeg" if image_path.endswith("jpg") else "image/png"
        response = safe_call(image_path, mime_type, metadata)

        if response is None:
            return None

        lis = response.split("\n")
        qas = list(csv.reader(lis))
        if not qas:
            return None

        if qas[0] == ['```csv']:
            del qas[0]
            del qas[-1]
        del qas[0]

        results = []
        for qa in qas:
            if qa:
                try:
                    results.append((image_path, qa[0], qa[1]))
                except IndexError:
                    pass
        return results
    except Exception as e:
        print(f"[ERROR] Processing index {i}: {e}")
        return None

if __name__ == "__main__":
    output_file = "main_image_qa_10.csv"
    max_threads = 8

    with open(output_file, "w", newline='', encoding="utf-8") as write_file:
        writer = csv.writer(write_file)
        futures = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for i, img_metadata in enumerate(images_metadata):
                futures.append(executor.submit(process_image, i, img_metadata))

            img_count = 0
            for idx, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    with lock:
                        for row in result:
                            writer.writerow([row[0], row[1], row[2]])
                        img_count += 1

                if (idx + 1) % 100 == 0:
                    print(f"[INFO] Images Processed: {idx + 1} / {len(futures)} | Images Used: {img_count}")
