from os.path import exists

from setuptools import setup, find_packages

#  install_requires = [
#  "torch >= 0.14.0",
#  "toolz >= 0.8.2",
#  "scikit-learn >= 0.18.0",
#  "scipy >= 0.13.0",
#  "numpy >= 1.8.0",
#  ]


def _get_version() -> str:
    with open("adadamp/__init__.py", "r") as f:
        raw = f.readlines()
    medium_rare = [l for l in raw if "__version__" in l]
    assert len(medium_rare) == 1
    medium = medium_rare[0].split(" = ")[1].strip()
    well_done = medium[1:-1]
    return well_done

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["numpy", "pandas"]


setup(
    name="adadamp",
    version=_get_version(),
    #  url=
    maintainer="Scott Sievert",
    maintainer_email="dev@stsievert.com",
    install_requires=install_requires,
    description="Batch size estimation for machine learning training",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
)
