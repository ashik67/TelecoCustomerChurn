from setuptools import setup, find_packages
from pathlib import Path
from typing import List

def read_requirements(file_path: str) -> List[str]:
    """Read the requirements from a file and return a list of packages."""
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip() and not line.startswith('-e,#')]
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No requirements will be installed.")
        return []
    

print("Reading requirements from requirements.txt...")
requirements = read_requirements('requirements.txt')

setup(
    name='TelecoCustomerChurn',
    version='0.1.0',
    author='Mohammed Ashik', 
    packages=find_packages(),
    install_requires=requirements,
)