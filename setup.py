from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


setup(
    name="autocatchem",
    version="0.0.1",
    author="Jason Hlozek",
    author_email="jason.hlozek@uct.ac.za",
    url="https://github.com/ersilia-os/autocat-chem",
    description="Automated Catboost modeling for chemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.7",
    install_requires=install_requires,
    packages=find_packages(exclude=("utilities")),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="catboost machine-learning chemistry computer-aided-drug-design",
    project_urls={
        "Source Code": "https://github.com/ersilia-os/autocat-chem",
    },
    include_package_data=True,
)
