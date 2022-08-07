import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="ssl_study",
    version="0.0.1",
    author="Piyush and Ayush",
    author_email="mein2work@gmail.com",
    description=(
        "Implementation of various self supervised learning methods and comparing their performance."
    ),
    license="GNU General Public License v3.0",
    keywords="ssl tensorflow keras",
    packages=["ssl_study", "tests", "configs"],
    long_description=read("README.md"),
)