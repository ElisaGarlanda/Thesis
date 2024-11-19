# Efficient Streaming Image Classification via Deep Embeddings and Random Projection
**MSc Thesis - Mathematical Engineering @ Politecnico di Milano**  

## Description
This repository contains all the code and resources required to reproduce the results of the MSc thesis:  
**Efficient Streaming Image Classification via Deep Embeddings and Random Projection**.

---

## Virtual environment
To set up the virtual environment:

1. Use the provided **`requirements.txt`** file to install Python libraries.
2. Install Java to ensure compatibility with dependencies required by CapyMOA.  
   For detailed installation instructions, refer to CapyMOA's official guide:  
   [CapyMOA Installation Guide](https://capymoa.org/installation.html)

---

## File descriptions
This repository includes the following Jupyter notebooks, which correspond to different stages of the pipeline:
- **`unzip_clear.ipynb`**   
  Unzip the dataset.

- **`feature_extraction.ipynb`**  
  Extract features from images using pre-trained CNNs.

- **`run_models.ipynb`**  
  Run streaming models with Random Projection applied after feature extraction.

- **`no RP.ipynb`**  
  Run streaming models without Random Projection after feature extraction.

Additionally, Python scripts in the repository provide utility functions called by the Jupyter notebooks. It is necessary to run **`feature_extraction.ipynb`** before running **`run_models.ipynb`** and **`no RP.ipynb`**.

---

## Pre-trained models
The repository includes pre-trained ResNet50 model parameters:

- **`state_dict.pth.tar`**: MoCo pre-training without additional fine-tuning.  
- **`finetuned_model_moco9.pth.tar`**: MoCo pre-training with fine-tuning.

---

## Dataset
The dataset used in this thesis can be downloaded from the following links:

- **CLEAR dataset official website**: [CLEAR Benchmark](https://clear-benchmark.github.io/)  
- **Training set**: [clear100-train-image-only.zip](https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-train-image-only.zip)  
- **Test set**: [clear100-test.zip](https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-test.zip)
