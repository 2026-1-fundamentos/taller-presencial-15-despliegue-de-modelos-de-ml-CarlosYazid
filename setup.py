"""Package installer"""

from setuptools import find_packages, setup  # type: ignore

setup(
    name="homework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "fastapi",
        "uvicorn",
        "pandas",
        "scikit-learn",
        "fg-data-profiling",
        "skl2onnx",
        "onnxruntime",
        "pydantic",
        "requests",
        "dash",
        "dash-bootstrap-components",
        "ipykernel",
        "ipywidgets"
    ],
)
