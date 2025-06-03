from setuptools import setup, find_packages

setup(
    name="openvino_training_wrapper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "tensorflow>=2.10",
        "scikit-learn",
        "openvino",
        "numpy",
    ],
    author="Leonardo Heim Monteiro",
    description="Training wrapper for AI PC using OpenVINO with PyTorch, TensorFlow and Scikit-learn",
    license="MIT",
    python_requires=">=3.8",
)
