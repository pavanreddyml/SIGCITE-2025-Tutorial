from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bsidesnova",
    version="0.0.1",
    description="",
    author="Pavan Reddy",
    author_email="pavan.reddy@gwmail.gwu.edu",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=parse_requirements("requirements.txt"),
)