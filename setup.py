import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="empirical",
    version="1.0.0",
    author="Quakecore",
    description="Package for empirical calculations",
    url="https://github.com/ucgmsim/Empirical_Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        "empirical": [
            "util/*.yaml",
        ]
    },
    scripts=[
        "empirical/scripts/calculate_empirical.py",
        "empirical/scripts/emp_aggregation.py",
        "empirical/scripts/aggregate_empirical_im_permutations.py",
    ],
)
