import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import gradio as gr
from PIL import Image # Required for image processing

# --- 1. Load MNIST Data ---
# This part loads the MNIST dataset. It's crucial for training your model.
print("Loading MNIST data...")
# fetch_openml downloads the dataset. as_frame=False returns NumPy arrays.
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
print("MNIST data loaded. X shape:", X.shape, "y shape:", y.shape)

# --- 2. Prepare Data for Binary Classification (Digit '1' vs. Not '1') ---
# We split the data into training and testing sets.
# MNIST has 70,000 samples; we use the first 60,000 for training and rest for testing.
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Create binary labels: True if the digit is '1', False otherwise.
# The model will learn to predict 1 for '1' and 0 for 'not 1'.
y_train_1 = (y_train == '1')
y_test_1 = (y_test == '1')
print("Data split and binary target '1' prepared.")

# --- 3. Train SGDClassifier Model ---
# SGDClassifier (Stochastic Gradient Descent Classifier) is a linear model.
# It's efficient for large datasets.
# 'class_weight='balanced'' is added to handle class imbalance,
# giving more importance to the minority class ('1's).
print("Training SGDClassifier for '1' vs 'not 1' classification...")
sgd_clf = SGDClassifier(
    random_state=42,           # For reproducibility of results.
    class_weight='balanced',   # Automatically adjusts weights inversely proportional to class frequencies.
    loss='log_loss',           # Use 'log_loss' for logistic regression style classification.
                               # Other options: 'hinge' (linear SVM), 'perceptron'.
    max_iter=1000,             # Increased max_iter. Default is 1000, ensure it converges.
    tol=1e-3,                  # Tolerance for the stopping criterion.
    early_stopping=True,       # Stop training when validation score does not improve.
    validation_fraction=0.1,   # The proportion of training data to set aside as validation set for early stopping.
    n_iter_no_change=5,        # Number of iterations with no improvement to wait before stopping.
    verbose=True               # Prints training progress (loss at each iteration).
)
sgd_clf.fit(X_train, y_train_1) # Train the model with the binary labels.
print("SGDClassifier training complete.")

# Print the accuracy of the trained model on both training and testing sets.
print("Training Accuracy (SGD): ", sgd_clf.score(X_train, y_train_1))
print("Testing Accuracy (SGD): ", sgd_clf.score(X_test, y_test_1))


# --- 4. Prediction Function for Gradio Interface ---
# This function takes the image drawn by the user and processes it for prediction.
def predict(img_input):
    """
    Predicts if the drawn image is a '1' or not using the trained SGDClassifier.

    Args:
        img_input (dict): The input from Gradio's ImageEditor component.
                          It contains the image data under the 'background' key.
                          Example: {'background': array([...], shape=(800, 800, 4), dtype=uint8)}
    Returns:
        int: 1 if the model predicts the digit is '1', 0 if it predicts 'not 1'.
             Returns a string message if the input is invalid or empty.
    """
    # Basic input validation: Check if an image was actually drawn/provided.
    if img_input is None or 'background' not in img_input or img_input['background'] is None:
        return "Please draw a digit on the canvas."

    # Extract the raw image data (NumPy array) from the 'background' key.
    img_array = img_input['background']

    # Ensure the extracted data is indeed a NumPy array.
    if not isinstance(img_array, np.ndarray):
        return "Error: Image background is not a NumPy array. Unexpected input format."

    # Convert the NumPy array (which is likely RGBA - Red, Green, Blue, Alpha channels)
    # into a PIL (Pillow) Image object. This is useful for image manipulations.
    # We cast to uint8 to ensure compatibility with PIL.
    pil_image = Image.fromarray(img_array.astype(np.uint8))

    # Convert the image to grayscale ('L' mode in PIL) and resize it to 28x28 pixels.
    # MNIST images are 28x28 grayscale, so this step is critical for matching the model's input expectation.
    img_processed = pil_image.convert('L') # 'L' mode means 8-bit pixels, black and white.
    img_processed = img_processed.resize((28, 28)) # Resize to the target dimensions.

    # Convert the processed PIL Image back into a NumPy array.
    img_processed = np.array(img_processed)

    # --- CRUCIAL FIX: Invert colors if the background is white ---
    # The SGDClassifier was trained on MNIST, where digits are white on a black background.
    # If the user draws on a white background (common in drawing apps), the digit will be black.
    # We check the top-left pixel (assumed to be background). If it's bright (e.g., > 128),
    # it means the background is white, and we need to invert the image.
    if img_processed[0, 0] > 128: # A threshold of 128 (mid-gray) is used to detect bright background.
        img_processed = 255 - img_processed # Invert pixel values (0 becomes 255, 255 becomes 0).

    # Normalize pixel values: scale them from 0-255 to 0.0-1.0.
    # This is a standard preprocessing step for neural networks.
    # Reshape the 28x28 image into a 1x784 (flattened) array, as expected by the SGDClassifier.
    img_final = img_processed.reshape(1, 784) / 255.0

    # Make a prediction using the trained SGDClassifier.
    # .predict() returns an array, so [0] extracts the single prediction.
    prediction = sgd_clf.predict(img_final)[0]

    # Return the prediction as an integer (1 or 0).
    return int(prediction)

# --- 5. Launch Gradio Interface ---
# This sets up the web-based user interface for your digit recognizer.
print("Launching Gradio interface...")
gr.Interface(
    fn=predict, # The function to call when the user submits an image.
    inputs=gr.ImageEditor( # The input component, allowing users to draw.
        type="numpy",      # Specifies that the image data should be passed as a NumPy array (within a dict).
        image_mode="L",    # Sets the image editor to grayscale mode, matching MNIST.
        width=280,         # Display width of the drawing canvas.
        height=280,        # Display height of the drawing canvas.
        label="Draw a digit (0-9)" # Label displayed above the drawing area.
    ),
    outputs=gr.Label(label="Prediction (1 if '1', 0 if 'not 1')"), # The output component to display the prediction.
    title="MNIST Digit Recognizer (SGDClassifier - Is it a '1'?)", # Title of the Gradio app.
    description="Draw a single digit (0-9) on the canvas. The model will predict '1' if it thinks it's the digit '1', and '0' otherwise." # Description for the app.
).launch(share=True) # Starts the Gradio interface.
