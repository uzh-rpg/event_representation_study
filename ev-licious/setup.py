from setuptools import setup, find_packages

setup(
    name='evlicious',
    version='0.1',
    packages=['evlicious',
              'evlicious/io',
              'evlicious/tools',
              'evlicious/art',
              'evlicious/io/utils',
              'evlicious/io/utils/dvs_msgs'],
    package_dir={'':'src'}
)
