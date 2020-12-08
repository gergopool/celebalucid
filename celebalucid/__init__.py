import os

from celebalucid.models.manipulator import ModelManipulator
from celebalucid.generator import build_generator

__version__ = "0.1.2"
__author__ = 'Gergely Papp'
__credits__ = 'Alfred Renyi Institute of Mathematics'

base_url = 'https://users.renyi.hu/~gergopool/lucid/'

def load_model(str_model):
    url = os.path.join(base_url, str_model+'.pt')
    return ModelManipulator(url)


