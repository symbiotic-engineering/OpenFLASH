from setuptools import setup, find_packages

# Read requirements from the file (from the root directory)
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='meem',  # Package name
    version='0.1.0',  # Package version
    description='A Python package for matched eigenfunctions methods',  # Short description
    long_description=open('README.md').read(),  # Read long description from README
    long_description_content_type='text/markdown',  # Set content type for markdown
    url='https://github.com/symbiotic-engineering/semi-analytical-hydro.git',  # GitHub URL
    packages=find_packages(where='package/src'),  # Find packages under 'package/src'
    package_dir={'': 'package/src'},  # Set the package root directory to 'package/src'
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Minimum Python version required
    install_requires=requirements,  # Install dependencies from requirements.txt
    license='MIT',  # License type
)
