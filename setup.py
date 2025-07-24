from setuptools import find_packages, setup
from typing import List

HYPER_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements.txt file and returns a list of dependencies.

    :param file_path: Path to the requirements.txt file
    :return: List of package requirements
    """
    requirements = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]  # Correct newline removal

    if HYPER_E_DOT in requirements:
        requirements.remove(HYPER_E_DOT)
        
    return requirements

setup(
    name="Credit_Card_Fraud_Detection",
    version="0.0.1",
    author="Dhruv Parmar",
    author_email='dhruvparmar051@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description="A machine learning project for detecting credit card fraud.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
