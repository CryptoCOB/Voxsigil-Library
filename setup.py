from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="voxsigil-library",
    version="2.0.0",
    description="OpenClawd-VoxBridge integration SDK - AI agents for prediction markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CryptoCOB",
    author_email="support@voxsigil.online",
    url="https://github.com/CryptoCOB/Voxsigil-Library",
    project_urls={
        "Homepage": "https://github.com/CryptoCOB/Voxsigil-Library#readme",
        "Documentation": "https://github.com/CryptoCOB/Voxsigil-Library",
        "Bug Reports": "https://github.com/CryptoCOB/Voxsigil-Library/issues",
        "Source": "https://github.com/CryptoCOB/Voxsigil-Library",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "cronos": [
            "eth-account>=0.8.0",
        ],
    },
    keywords=[
        "molt-agent",
        "voxsigil",
        "openclawd",
        "voxbridge",
        "prediction",
        "agent",
        "markets",
        "ai-agent",
        "coordination",
        "cronos",
        "eip-191",
        "blockchain"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
