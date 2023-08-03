from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name="swafi",
    version="0.0.1",
    author="Pascal Horton",
    author_email="pascal.horton@unibe.ch",
    description="Surface Water Flood Impact",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=['swafi',
              'swafi.utils'],
    package_dir={'swafi': 'swafi',
                 'swafi.utils': 'swafi/utils',
                 },
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    readme="README.md",
    project_urls={
        "Source Code": "https://github.com/pascalhorton/surface-water-flood-impact",
        "Bug Tracker": "https://github.com/pascalhorton/surface-water-flood-impact/issues",
    },
    license="MIT",
)
