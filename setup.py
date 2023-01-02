from setuptools import setup

setup(name = "mammoth",
      version = '1.0',
      author = 'RanDropper',
      author_email= "randropper001@gmail.com",
      url = "https://github.com/RanDropper/mammoth",
      description = "An advanced time series prediction framework.",
      packages = ["mammoth"],
      download_url = "https://github.com/RanDropper/mammoth.git",
      install_requires = ["tensorflow>=2.3.0"])