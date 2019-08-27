import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uniblock",
    version="1.0.4",
    author='Ringo',
    description=(
        "uniblock, scoring and filtering corpus"
        " with Unicode block information (and more)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ringoreality/uniblock",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'joblib',
        'sklearn',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['uniblock=uniblock.uniblock:main'],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
