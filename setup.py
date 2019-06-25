from os import path
from setuptools import setup


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="fhi_fed_utils",
    version="0.0.1",
    author="Helene Seiler",
    author_email="seiler.helene@gmail.com",
    description="A set of functions for fed analysis.",
    license="AGPL3",
    keywords="fed",
    url="https://github.com/hseiler/fhi_fed_utils",
    packages=[],
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3.7',
    ],
    py_modules=["fhi_fed_utils"],
    python_requires='>=3.4, <4',
    install_requires=[
        'scipy',
        'numpy'
    ],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/hseiler/fhi_fed_utils/issues',
        'Source': 'https://github.com/hseiler/fhi_fed_utils',
    },
)
