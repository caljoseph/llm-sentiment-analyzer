from setuptools import setup, find_packages

setup(
    name="llm_sentiment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "litellm>=1.45.0",
        "torch>=2.0.0",
        "pandas>=1.0.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.0.0",
        "pydantic>=2.0.0"
    ],
    entry_points={
        "console_scripts": [
            "llm-sentiment=llm_sentiment.cli.main:main",
        ],
    },
    python_requires=">=3.8",
)
