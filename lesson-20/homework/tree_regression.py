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
        """
        This is a recursive function to build the regression tree.
        """
        num_samples = X.shape[0]

        # Stop conditions:
        # 1. Max depth reached
        # 2. Number of samples below minimum split threshold
        # 3. All target values in the node are the same (perfect split, no further reduction possible)
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
        """
        Calculates the Mean Squared Error (MSE) for a given set of target values.
        For regression, MSE is used as the impurity measure.
        """
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y))**2)

    def _best_split(self, X, y):
        """
        Finds the best split (feature and threshold) that maximizes information gain
        (reduction in MSE) for regression.
        """
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

                # Ensure that both splits have enough samples
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                left_mse = self._mse(y[left_mask])    # MSE of left child
                right_mse = self._mse(y[right_mask])  # MSE of right child

                # Weighted average MSE of child nodes
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
        """
        Fits the Decision Tree Regressor to the training data.
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.tree = self._build_tree(self.X, self.y)

    def _predict_one(self, x, tree):
        """
        Recursively traverses the tree to predict the value for a single sample.
        """
        if tree["leaf"]:
            return tree["value"] # Return the predicted value for leaf node
        
        # Determine which branch to follow based on the feature and threshold
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        """
        Predicts target values for new data.
        """
        return np.array([self._predict_one(x, self.tree) for x in np.array(X)])

# --- Testing the DecisionTreeRegressor ---

# Generate a synthetic regression dataset
X_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)

# Split the data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Initialize and train the DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
regressor_tree.fit(X_train_reg, y_train_reg)

# Make predictions on the test set
predictions_reg = regressor_tree.predict(X_test_reg)

# Evaluate the model
mse = mean_squared_error(y_test_reg, predictions_reg)
r2 = r2_score(y_test_reg, predictions_reg)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# You can also print a few predictions vs actual values
print("\nSample Predictions vs Actual Values:")
for i in range(5):
    print(f"Predicted: {predictions_reg[i]:.2f}, Actual: {y_test_reg[i]:.2f}")