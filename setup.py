from setuptools import setup, find_packages

setup(
    name="research_agent",
    version="1.0.0",
    packages=find_packages(include=[
        "app",
        "llms",
        "cache",
        "graph",
        "agents",
        "memory",
        "tools"
    ]),
)