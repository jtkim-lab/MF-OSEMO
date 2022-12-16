from setuptools import setup

path_requirements = 'requirements.txt'
list_packages = [
    'mf_osemo',
]

with open(path_requirements) as f:
    required = f.read().splitlines()

setup(
    name='mf_osemo',
    version='0.1.0',
    license='MIT',
    packages=list_packages,
    python_requires='>=3.6, <4',
    install_requires=required,
)
