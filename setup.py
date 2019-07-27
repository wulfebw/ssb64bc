from setuptools import setup

setup(name="ssb64bc",
      version="0.1",
      description="A package for training behavioral cloning agents of ssb64.",
      author="Blake Wulfe",
      author_email="blake.w.wulfe@gmail.com",
      license="MIT",
      packages=["ssb64bc"],
      zip_safe=False,
      install_requires=[
        "numpy",
        "pandas"
      ])
