import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression # For testing regression

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _build_tree(self, X, y, depth=0):
        num_samples = X.shape[0]

        if (self.max_depth is not None and depth >= self.max_depth) or \
           num_samples < self.min_samples_split or \
           np.all(y == y[0]): # Check if all y values are identical
            leaf_value = np.mean(y) # For regression, leaf value is the mean of target values
            return {
                "leaf": True,
                "value": leaf_value # Use 'value' instead of 'class' for regression
            }

        # Find the best split
        feature_idx, threshold, gain = self._best_split(X, y)

        # If no gain or gain is non-positive, make it a leaf node
        if gain <= 0:
            leaf_value = np.mean(y)
            return {
                "leaf": True,
                "value": leaf_value
            }

        # Split the data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y))**2)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx = None
        split_threshold = None

        parent_mse = self._mse(y) # Calculate MSE of the parent node
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                left_mse = self._mse(y[left_mask])    
                right_mse = self._mse(y[right_mask]) 
              
                n = y.shape[0]
                child_mse = (left_mask.sum() / n) * left_mse + \
                            (right_mask.sum() / n) * right_mse

                # Gain is the reduction in MSE
                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold
        return split_idx, split_threshold, best_gain

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.tree = self._build_tree(self.X, self.y)

    def _predict_one(self, x, tree):
        """
        Recursively traverses the tree to predict the value for a single sample.
        """
        if tree["leaf"]:
            return tree["value"]
        
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in np.array(X)])

# Generate a synthetic regression dataset
np.random.seed(42)
X = np.random.rand(200, 1) - 0.5  # a single random input feature
y = X ** 2 + 0.025 * np.random.randn(200, 1)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
regressor_tree.fit(X_train, y_train)

# Make predictions on the test set
predictions_reg = regressor_tree.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions_reg)

print(f"Mean Squared Error (MSE): {mse:.2f}")
