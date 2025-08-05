import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y = np.array([0, 0, 0, 1, 1, 2, 1, 0])
classes, counts = np.unique(y, return_counts = True)
print(classes, counts)
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _build_tree(self, X, y, depth = 0):
        """
        This is recursive function to build tree.
        """
        num_samples = X.shape[0]
        num_classes = len(np.unique(y))
        # Stop condition
        if(self.max_depth and depth >= self.max_depth) or num_classes ==1 or num_samples< self.min_samples_split:
            leaf_value = np.bincount(y).argmax()
            return {
                "leaf" : True, 
                "class" : leaf_value
            }

        # Find best split
        feature_idx, threshold, gain = self._best_split(X, y)

        if(gain<=0):
            leaf_value = np.bincount(y).argmax()
            return {
                "leaf" : True, 
                "class" : leaf_value
            }
        
        # Split
        left_mask = X[:, feature_idx]<=threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth+1)

        return {
                "leaf" : False, 
                "feature" : feature_idx,
                "threshold" : threshold,
                "left" : left_subtree,
                "right" : right_subtree
            }

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts = True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs**2)  

    def _best_split(self, X, y):

        best_gain = -1
        split_idx = None
        split_threshold = None

        parent_gini = self._gini(y)
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature]<=threshold
                right_mask = ~left_mask 

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                # Weighted average gini 
                n = y.shape[0]
                child_gini = (left_mask.sum()/n)*left_gini+(right_mask.sum()/n)*right_gini

                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold 
        return split_idx, split_threshold, best_gain

    
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        self.tree = self._build_tree(X, y)

    # This is also recursive function
    def _predict_one(self, x, tree):
        if(tree["leaf"]):
            return tree["class"]
        if x[tree["feature"]]<= tree["threshold"]:
            return self._predict_one(x, tree["left"])
        else: 
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in np.array(X)])

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X = iris.data[['petal length (cm)', 'petal width (cm)']].values 
y=iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth = 2, random_state = 42)
tree_clf.fit(X_train,y_train)

pred1= tree_clf.predict(X_test)

tree_clf2 = DecisionTreeClassifier(max_depth = 2)
tree_clf2.fit(X_train,y_train)      

pred2 = tree_clf2.predict(X_test)

print(accuracy_score(y_test, pred1))
print(accuracy_score(y_test, pred2))
print(pred1)
print(pred2)

