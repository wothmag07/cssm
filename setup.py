from setuptools import setup, find_packages

setup(
    name="e-commerce-backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain_core",
        "langchain_astradb",
        "fastapi",
        "uvicorn",
        "langchain",
    ],
    author="Gowtham Arulmozhi",
    author_email="gowtham.arulmozhi@gmail.com",
    description="Customer support system Management",
)
