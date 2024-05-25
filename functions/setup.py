from setuptools import setup, find_packages

setup(
    name='inductive-oocr-functions',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list your project's dependencies here, e.g.,
        'munch==4.0.0',
        'numpy==1.26.4',
        'openai==1.30.2',
        'PyYAML==6.0.1',
        'scipy==1.13.1',
        'tiktoken==0.7.0',
        'tqdm==4.66.4',
    ],
    # PyPI metadata
)