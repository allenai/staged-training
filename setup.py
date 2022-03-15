import setuptools


with open('requirements.txt') as f:
    required = f.read().splitlines()


setuptools.setup(
    name='doubling',
    version='0.0.1',
    url='https://github.com/allenai/doubling',
    packages=setuptools.find_packages(),
    install_requires=required,
)
