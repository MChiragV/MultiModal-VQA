# Visual Recognition Mini Project: Visual Question Answering(VQA)
## Directory Structure
```
Baseline Evaluation/
    ├── BLIP.ipynb
    ├── LLaVA.ipynb
    ├── Qwen.ipynb
    └── vilbert.ipynb
Data-Curation/
    ├── Data Preparation.ipynb
    ├── data_prep_multithreaded.py
    ├── data_prep_ollama.py
    ├── data_prep_requirements.txt
    ├── data_prep.py
    ├── dataset-20k.csv
    ├── dataset-6k.csv
    └── utils.py
Fine-tuning scripts/
    ├── blip.ipynb
    ├── blip2.ipynb
    ├── llava.ipynb
    ├── microsoft-git.ipynb
    ├── phi3.ipynb
    └── qwen.ipynb
inference_setup/
    ├── inference.py
    └── requirements.txt
dataset-20k.csv
dataset-6k.csv
README.md
VR_Mini_Project_2_Report.pdf
```
## Data-Curation
This directory contains:
- Data Preparation.ipynb: Dataset generation using Google Developers API
- utils.py: Some helper functions for filtering the metadata
- data_prep.py: Dataset generation using Vertex AI API (single thread)
- data_prep_multithreaded.py: Dataset generation using Vertex AI API (multi-threaded)
- data_prep_requirements.txt: Required packages to run all the file
- data_prep_ollama.py: Dataset generation using Ollama

## Baseline Evaluation
This directory contains 4 notebooks:
- BLIP.ipynb: This notebook is used to run the BLIP pretrained model.
- LLaVA.ipynb: This notebook is used to run the LLaVA pretrained model.
- Qwen.ipynb: This notebook is used to run the Qwen pretrained model.
- VilBERT.ipynb: This notebook is used to run the VilBERT pretrained model.

## Fine-tuning scripts
This directory contains 6 notebooks:
- blip.ipynb: This notebook is used to run the BLIP fine-tuned model.
- blip2.ipynb: This notebook is used to run the BLIP2 fine-tuned model.
- llava.ipynb: This notebook is used to run the LLaVA fine-tuned model.
- microsoft-git.ipynb: This notebook is used to run the Microsoft Git fine-tuned model.
- phi3.ipynb: This notebook is used to run the Phi3 fine-tuned model.
- qwen.ipynb: This notebook is used to run the Qwen fine-tuned model.

## inference_setup
Please download the files from this directory to test our best fine-tuned model.

- inference.py: This file is used to run our best fine-tuned model on given data.
- requirements.txt: This file contains the requirements to run the inference.py file.

## Outside files
Other than the above directories, there are 3 files(except README.md):

- dataset-6k.csv: This file contains the dataset of 6k images and their corresponding questions and answers which is used to fine-tune the models.
- dataset-20k.csv: This file contains the dataset of 20k images and their corresponding questions and answers which is used to fine-tune the models.
- VR_MiniProject_2_Report.pdf: This file contains the report of the project 
