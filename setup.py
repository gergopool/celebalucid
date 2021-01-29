from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

setup(
    name='celebalucid',
    version='0.1.5.3',
    author='Gergely Papp',
    author_email='gergopool@gmail.com',
    packages=find_packages(),
    package_dir={'celebalucid': 'celebalucid'},
    package_data={'celebalucid' : ['res/*.txt']},
    url='http://pypi.python.org/pypi/celebalucid/',
    license='LICENSE',
    description='Analysing Clarity\'s InceptionV1 network after transfer learning on CelebA dataset.',
    install_requires=requirements,
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)