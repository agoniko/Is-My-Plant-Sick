# Plant Disease Classification
Increasing Models robustness with Combined Datasets, Data Augmentation and Insightful Attention Map Visualization.

## Description
This project aims to increase the models robustness and generalization capacity when it comes to plant diseases classification using the PlantVillage dataset.

Since the dataset has a very specific type of images (single leaf at the center with a clear background) and has a huge bias when it comes to capturing conditions (as mentioned at this [link](https://towardsdatascience.com/uncovering-bias-in-the-plantvillage-dataset-7ac564334526#:~:text=This%20indicates%20significant%20dataset%20bias,49%25%20accuracy%20using%20pure%20noise)) a combination of PlantVillage and PlantDoc dataset is made. 
PlantDoc is a plant disease detection dataset so, firstly leaf images are extracted from each bounding box, then the two datasets are combined. The selected classes are the ones in common between the two datasets.

Resnet and ViT models are trained on the resulting dataset.

A performance comparison is firstly done on Resnet and ViT models on the unified dataset, then, a ViT trained only on the PlantVillage dataset and one trained on the unified dataset are compared on images scraped from the web to assess if robustness and generalization capacity has increased.
The scraped images are similar to the ones a typical user might upload when using the service. They are also very different from the training ones since they can contain more than one leaf.

## Platform
To use the models a platform with streamlit is been developed. Here you can see how predictions are made on the test set also visualizing attention maps. There is also a specific section where you can upload your damaged leaf photo and see how to treat it as well as damaged areas seen by the model.

## Installation
Root directory contains a requirements.txt file, simply type `pip install -r requirements.txt` in your preferred env.

## Notebooks usage
To lighten the project folder I have included only the unified dataset so the following notebooks are <b>NOT</b> runnable:
```
notebooks/EDA-PlantVillage.ipynb
notebooks/synthetic-PlantDoc.ipynb
notebooks/dataset_unification.ipynb
```
while for the notebooks:
```
notebooks/training_and_evaluation.ipynb
notebooks/attention_visualization.ipynb
```
firstly change the AI accelerator device, the variable containing it's reference is in the top cells after the library imports, then you can simply press the <b> Run All </b> button (training lines are commented).

## Platform Usage
The platform implemented with streamlit also needs a FastAPI backend running. That is because a custom image gallery grid has been created with some javascript code.

To run it open a new terminal in the project folder:
```bash
cd app/backend

uvicorn requests_handler:app --reload
```

Now open another terminal still in the project folder:
```bash
cd app
streamlit run Home_page.py
```

Enjoy 😃