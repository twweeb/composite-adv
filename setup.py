import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="composite_adv",
    version="0.0.1",
    author="Lei Hsiung",
    author_email="hsiung@m109.nthu.edu.tw",
    description="Composite Adversarial Attack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twweeb/composite-adv",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.4.0',
        'numpy>=1.16.0',
        'torchvision>=0.5.0',
        'kornia>=0.6.1',
        'sinkhorn-knopp>=0.2',
        'requests>=2.23.0',
        'advex-uar>=0.0.5.dev0',
        'scipy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)