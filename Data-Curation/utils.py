import os
import pandas as pd

IMAGES_BASE_DIR = os.path.join("abo-images-small", "images")
IMAGES_PATH = os.path.join(IMAGES_BASE_DIR, "small")
IMAGES_METADATA_PATH = os.path.join(IMAGES_BASE_DIR, "metadata")
IMAGES_METADATA_CSV = os.path.join(IMAGES_METADATA_PATH, "images.csv")

LISTINGS_METADATA_PATH = os.path.join("abo-listings", "listings", "metadata")

def get_main_image_path(metadata, df):
    img_id = metadata["main_image_id"]
    img_entry = df[df["image_id"] == img_id]
    img_path = os.path.join(IMAGES_PATH, img_entry["path"].iloc[0])

    return img_path

def get_other_image_paths(metadata, df):
    img_ids = metadata["other_image_id"]
    img_paths = []
    for img_id in img_ids:
        img_entry = df[df["image_id"] == img_id]
        img_path = os.path.join(IMAGES_PATH, img_entry["path"].iloc[0])
        img_paths.append(img_path)

    return img_paths

def get_required_info(metadata):
    keys1 = {
                "bullet_point" : "Bullet Points", 
                 "color" : "Colour", 
                 "fabric_type" : "Fabric Type", 
                 "finish_type" : "Finish Type", 
                 "item_keywords" : "Item Keywords", 
                 "item_name" : "Item Name", 
                 "item_shape" : "Item Shape",
                "material" : "Material", 
                 "pattern" : "Pattern", 
                 "product_description" : "Product Description", 
                 "style" : "Style"
            } # Format: [{ "language_tag": <str>, "value": <str> }, ...]
    keys2 = {"color" : "Colours"} # Format: [{"language_tag": <str>, "standardized_values": [<str>],"value": <str>}, ...]
    keys3 = {"product_type" : "Product Type"} # Format: <str>

    strings = []


    for key in keys1.keys():
        lis = metadata.get(key, [])
        vals = set()
        for entry in lis:
            lang = entry["language_tag"]
            val = entry["value"]
            if lang.lower().startswith("en"):
                vals.add(val)
        if vals:
            strings.append(keys1[key] + ": " + ", ".join(vals))

    for key in keys2.keys():
        lis = metadata.get(key, [])
        vals = set()
        for entry in lis:
            lang = entry["language_tag"]
            val = entry["value"]
            std_vals = entry.get("standardized_values", [])
            if lang.lower().startswith("en"):
                vals.add(val)
                vals = vals | set(std_vals)
        if vals:
            strings.append(keys2[key] + ": " + ", ".join(vals))

    for key in keys3.keys():
        lis = metadata.get(key, [])
        vals = set()
        for entry in lis:
            val = entry["value"]
            vals.add(val)
        if vals:
            strings.append(keys3[key] + ": " + ", ".join(vals))

    return strings  