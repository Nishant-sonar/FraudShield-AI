from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    """ This function will return the list of requirements """
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="aml_fraud_detector",
    version="2.1.0",
    author="Nishant Sonar",
    author_email="nishantsonar047@gmail.com",
    description="Professional Anti-Money Laundering (AML) Fraud Detection System with ML and Rule-Based Analysis",
    long_description="A comprehensive fraud detection system combining machine learning models with rule-based analysis for financial transaction assessment. Features include SHAP explainability, risk scoring, and professional Streamlit dashboard.",
    long_description_content_type="text/plain",
    url="https://github.com/Nishant-sonar/FraudShield-AI",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(), # It looks for __init__.py files in subdirectories to identify valid Python packages.
                             # and Automatically detects and includes all Python packages in the project directory. 
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8",

)