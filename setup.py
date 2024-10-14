from setuptools import setup, find_packages

setup(
    name="ocular-streamlining",
    version="0.1.0",
    packages=find_packages(include=['data', 'models', 'stream', 'reports', 'notebooks', 'references', 'streamlining_training']),
    # other setup arguments
)