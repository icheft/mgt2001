import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mgt2001",
    version="0.0.13",
    author="Brian L. Chen",
    author_email="brian.lxchen@gmail.com",
    description="A small package for MGT 2001 use only",
    long_description_content_type="text/markdown",
    url="https://github.com/icheft/MGT2001_package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','pandas', 'matplotlib', 'adjustText'],
    include_package_data=True,
)