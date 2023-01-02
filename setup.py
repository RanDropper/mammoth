from setuptools import setup

setup(name = "mammoth",
      version = '1.0',
      author = 'RanDropper',
      author_email= "randropper001@gmail.com",
      url = "https://github.com/RanDropper/mammoth",
      description = "An advanced time series prediction framework.",
      packages = ["mammoth", "mammoth.model", "mammoth.networks"],
      download_url = "https://github.com/RanDropper/mammoth.git",
      install_requires = ["tensorflow>=2.3.0", "numpy>=1.10.1", "pandas>=0.25.1"])