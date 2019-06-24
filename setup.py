import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="fhi-fed-utils",
    version="0.0.1",
    author="Helene Seiler",
    author_email="seiler.helene@gmail.com",
    description=("A set of functions for fed analysis."),
    license="AGPL3",
    keywords="fed",
    # url="http://packages.python.org/an_example_pypi_project",
    packages=[],
    long_description=read('README'),
    install_requires=[
        'scipy',
        'numpy'
    ],
)
