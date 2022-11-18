from setuptools import setup, find_packages
import os


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='liana',
      version=get_version("liana/__init__.py"),
      description='LIANA - a LIgand-receptor ANalysis frAmework',
      author='Daniel Dimitrov',
      author_email='daniel.dimitrov@uni-heidelberg.de',
      url='https://github.com/saezlab/liana-py',
      project_urls={
          "Bug Tracker": "https://github.com/saezlab/liana-py/issues",
      },
      packages=find_packages(),
      package_data={'liana': ['resource/omni_resource.csv']},
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires=["tqdm",
                        "pandas",
                        "anndata",
                        "scanpy",
                        "plotnine"
                        ],
      python_requires=">=3.7",
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent"]
      )

