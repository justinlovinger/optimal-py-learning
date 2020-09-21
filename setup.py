from distutils.core import setup

# Convert README.md to long description
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r", "")  # YOU NEED THIS LINE
except (ImportError, OSError, IOError):
    print("Pandoc not found. Long_description conversion failure.")
    import io
    # pandoc is not installed, fallback to using raw contents
    with io.open('README.md', encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='learning',
    version='0.1.0',
    packages=['learning', 'learning.architecture', 'learning.data', 'learning.optimize', 'learning.testing'],
    # Include examples and datasets
    package_data={'learning': ['examples/*.py', 'data/datasets/*.data']},
    # Dependencies
    install_requires=[
        'numpy'
    ],

    # Metadata
    author='Justin Lovinger',
    license='MIT',
    description="A python machine learning library, with powerful customization for advanced users, and robust default options for quick implementation.",
    long_description=long_description,
    keywords=['machine-learning', 'supervised-learning', 'mulilayer-perceptron', 'mlp', 'neural-network', 'rbf-network', 'ensemble-learning', 'self-organizing-map', 'som', 'optimization', 'gradient-descent', 'linear-regression', 'regression', 'l-bfgs'],

    url='https://github.com/justinlovinger/learning',
)
