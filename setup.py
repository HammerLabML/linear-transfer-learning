import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="em_transfer_learning",
    version="1.0.0",
    author="Benjamin Paassen",
    author_email="bpaassen@techfak.uni-bielefeld.de",
    description="Supervised linear transfer learning based on labelled Gaussian mixture models and expectation maximization in scikit-learn-compatible form.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ub.uni-bielefeld.de/bpaassen/transfer-learning",
    packages=['em_transfer_learning'],
    install_requires=['numpy', 'scikit-learn', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords='transfer-learning gaussian-mixture-models learning-vector-quantization gmm lvq',
)
