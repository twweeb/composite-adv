import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="composite_adv",
    version="0.0.1",
    author="Lei Hsiung",
    author_email="nthu.email@gmail.com",
    description="Composite Adversarial Attack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twweeb/composite-adv",
    packages=setuptools.find_packages(),
    install_requires=[
        'kornia>=0.6.1',
        'sinkhorn-knopp>=0.2',
        'torch>=1.8.1',
        'torchvision>=0.9.1',
        'scipy>=1.6.3',
        'numpy>=1.20.3',
        'requests>=2.27.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)