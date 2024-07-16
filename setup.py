from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="genai_toolbox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Nick Miller",
    author_email="miller.nick.c@gmail.com",
    description="A toolbox for generative AI tasks",
    long_description="A comprehensive toolbox for various generative AI tasks including transcription, diarization, and text prompting.",
    url="https://github.com/nickcmiller/genai_toolbox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)