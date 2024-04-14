from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str)->List[str]:
    """Returns the list of requirements

    Args:
        file_path (str): file path of requirements.txt

    Returns:
        List[str]: extracted list from requirements.txt
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements
            

setup(
    name = "Anime_Recommender",
    version = "0.0.1",
    author = "Amol",
    author_email = "amol.shrivastava22@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
    
    )