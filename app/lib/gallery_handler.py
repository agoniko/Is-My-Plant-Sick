from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
import cv2

sys.path.append("resources")
sys.path.append("datasets/unified")
from ai_engine import AIEngine

data_path = "../datasets/unified"
# data_path = "../Plant_without_background/"
disease_csv = "./resources/disease_description.csv"
csv_path = "./resources/labels.csv"


class GalleryHandlerMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def get_valid_indices(dataset, df, only_test=True, leaf_type=None, disease_type=None):
    background_indices = np.where(np.array(dataset.targets) == 3)[0]
    if only_test:
        _, indices = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42
        )
    else:
        indices = list(range(len(dataset)))

    indices = [i for i in indices if i not in background_indices]
    if leaf_type is not None:
        indices = [i for i in indices if df.loc[i]["leaf"] == leaf_type]
    if disease_type is not None:
        indices = [i for i in indices if df.loc[i]["disease"] == disease_type]

    return indices


class GalleryHandler(metaclass=GalleryHandlerMeta):
    def __init__(self, only_test=False):
        self.only_test = only_test
        self.dataset = ImageFolder(data_path)
        self.descriptions = pd.read_csv(disease_csv)
        self.labels_df = pd.DataFrame(
            [self.dataset.classes[i] for i in self.dataset.targets], columns=["class"]
        )
        self.labels_df["leaf"] = self.labels_df["class"].apply(
            lambda x: x.split("___")[0]
        )
        self.labels_df["disease"] = self.labels_df["class"].apply(
            lambda x: x.split("___")[1]
        )
        self.indices = get_valid_indices(self.dataset, self.labels_df, self.only_test)

        # I need to store classes with background since the model predicts also this class
        self.labels_df = self.labels_df.iloc[self.indices]

        self.leaf_classes = self.labels_df["leaf"].unique()
        self.disease_classes = self.labels_df["disease"].unique()

        self.model = AIEngine()
        assert self.indices == self.labels_df.index.tolist()

    def set_index(self, index):
        if index < 0 or index >= len(self.indices):
            raise Exception("Index out of bounds")
        else:
            self.index = index

    def get_image(self, index):
        image, _ = self.dataset[self.indices[index]]
        image, label, score, attention = self.model.predict(image)

        pred = self.dataset.classes[label]
        leaf, disease = pred.split("___")
        true = self.labels_df.loc[self.indices[index]][["class"]][0]
        true_leaf, true_disease = true.split("___")
        correct = (False, False)
        if leaf == true_leaf:
            correct = (True, correct[1])
        if disease == true_disease:
            correct = (correct[0], True)

        if disease == "healthy":
            symptoms, treatment = "", ""
        else:
            symptoms, treatment = self.descriptions[
                self.descriptions["disease"] == disease
            ].values[0][1:]

        return image, attention, symptoms, treatment, leaf, disease, correct, score

    def get_images_path(self):
        return [self.dataset.imgs[i][0] for i in self.indices]

    def get_leaf_types(self):
        # return leaf types for the rows in indices
        return ["All"] + list(self.labels_df["leaf"].unique())

    def get_disease_types(self):
        return ["All"] + list(self.labels_df.loc[self.indices]["disease"].unique())

    def filter(self, leaf_type, disease_type):
        if leaf_type == "All":
            leaf_type = None
        if disease_type == "All":
            disease_type = None
        self.indices = get_valid_indices(
            self.dataset,
            self.labels_df,
            only_test=self.only_test,
            leaf_type=leaf_type,
            disease_type=disease_type,
        )
