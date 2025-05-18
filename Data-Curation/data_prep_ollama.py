# I do not know if this works or not. I could not run Ollama in my laptop.

import base64
import requests
import uuid
from utils import *
import csv
import json

df = pd.read_csv(IMAGES_METADATA_CSV)

json_files = os.listdir(LISTINGS_METADATA_PATH)

json_file_paths = [os.path.join(LISTINGS_METADATA_PATH, filename) for filename in json_files]

images_metadata = []
for filepath in json_file_paths:
    print(f"[INFO] Reading {filepath}")
    with open(json_file_paths[1], mode="r", encoding="utf-8") as read_file:
        json_strings = read_file.readlines()
        for json_string in json_strings:
            images_metadata.append(json.loads(json_string))


# Encode the image as base64
def encode_image(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Unique session ID for maintaining context
session_id = str(uuid.uuid4())

# Conversation history
messages = []

# Add an image and prompt
def send_prompt_with_image(prompt, image_path=None):
    global messages

    images = [encode_image(image_path)] if image_path else []

    # Add user message
    messages.append({
        "role": "user",
        "content": prompt,
        "images": images
    })

    # Send request
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llava",
        "messages": messages
    }, stream=True)

    # Capture streamed response
    full_response = ""
    for chunk in response.iter_lines():
        if chunk:
            part = eval(chunk.decode("utf-8"))["message"]["content"]
            full_response += part
            # print(part, end='', flush=True)

    # Add model's response to the conversation history
    messages.append({
        "role": "assistant",
        "content": full_response
    })

    return full_response

# --- USAGE EXAMPLE ---

# First turn with image
prompt = "I am preparing a dataset to train a Visual Question Answering (VQA) model. \
    I have a set of images and corresponding metadata from Amazon product listings (Amazon Berkeley Object Dataset). \
    Using the image generate questions with single unambiguous one-word answers in English which should be answerable by \"SOLELY\" looking at the image without providing any other data.\
    You can use and its metadata to verify the answers. \
    Each question has to be independent of the others.\
    Since I need to automate parsing these question and answers, please provide them in CSV format: question, answer.\
    Please do not generate anything else other than question and answers as it makes it difficult to write an automated parser."

image_metadata = images_metadata[0]
req_info = get_required_info(image_metadata)
image_path = get_main_image_path(image_metadata, df)
metadata = "\n".join(req_info)
res = send_prompt_with_image(prompt + "\n" + metadata, image_path)
print(res)

# with open("main_image_qa.csv", "w") as write_file:
#     for i, img_metadata in enumerate(images_metadata):
#         if (i+1) % 100 == 0:
#             print(f"INFO: {(i+1)}/{len(images_metadata)}")

#         try:
#             req_info = get_required_info(img_metadata)
#             if len(req_info) < 5:
#                 continue

#             metadata = "\n".join(req_info)
#             image_path = get_main_image_path(img_metadata, df)
#             mime_type = "image/jpeg" if image_path.endswith("jpg") else "image/png"
#             response = call_with_inline(client, image_path, mime_type, metadata)

#             lis = response.split("\n")
#             qas = list(csv.reader(lis))
#             if not qas:
#                 continue

#             if qas[0] == ['```csv']:
#                 del qas[0]
#                 del qas[-1]
#             del qas[0]

#             if qas:
#                 for qa in qas:
#                     if not qa:
#                         continue
#                     write_file.write(f"{image_path}, {qa[0]}, {qa[1]}\n")
#         except Exception as e:
#             print("Error in inline approach:", e)

# Follow-up question (without re-uploading image)
# print("\n\nFollow-up:")
# send_prompt_with_image("What might this person be doing?")
