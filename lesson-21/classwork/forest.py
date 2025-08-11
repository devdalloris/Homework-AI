import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Literal

class RandomForestClassifier:
    def __init__(self, n_estimators: int = 10, max_depth: int|None =None, min_samples_split: int = 2, max_features: float|Literal = ['auto', 'sqrt', 'log2']|None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        
