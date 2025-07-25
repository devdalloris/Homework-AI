import gradio
import os

# This will show you the main directory where gradio is installed
gradio_path = os.path.dirname(gradio.__file__)
print(f"Gradio installation path: {gradio_path}")

# Now, let's check for the templates and static subfolders
templates_path = os.path.join(gradio_path, 'templates')
static_path = os.path.join(gradio_path, 'static')

print(f"Expected templates path: {templates_path}")
print(f"Expected static path: {static_path}")

# You can also verify if they exist
print(f"Templates path exists: {os.path.exists(templates_path)}")
print(f"Static path exists: {os.path.exists(static_path)}")