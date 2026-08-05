import pandas as pd
import warnings

from config import DATA_PATH
from feature_config import FEATURE_CONFIG

TARGET = "SalePrice"
def load_data():
    return pd.read_csv(DATA_PATH)
def get_features():
    df = load_data()
    return [
        col for col in df.columns
        if col != TARGET
    ]

def get_feature_types():
    df = load_data()
    numerical_features = []
    categorical_features = {}
    for feature in get_features():
        if df[feature].dtype == "str":
            categorical_features[feature] = (
                ["None"] +
                sorted(
                    df[feature]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )
        else:
            numerical_features.append(feature)
    return numerical_features, categorical_features

def create_sections():
    features = get_features()
    sections = {}
    for feature in features:
        if feature not in FEATURE_CONFIG:
            continue
            section = "Other Features"
        else:
            section = FEATURE_CONFIG[feature]["section"]
        if section not in sections:
            sections[section] = []
        sections[section].append(feature)


    return sections