from setuptools import setup, find_packages

setup(
    name="meter",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="0.1.0",
    license="MIT",
    description="METER: Multimodal End-to-end TransformER",
    author="Microsoft Corporation",
    author_email="zdou0830@gmail.com",
    url="https://github.com/zdou0830/METER",
    keywords=["vision and language pretraining"],
    install_requires=["torch", "pytorch_lightning"],
)
