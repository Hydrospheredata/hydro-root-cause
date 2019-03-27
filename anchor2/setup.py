from setuptools import setup

setup(name='anchor2',
      version='0.0.1',
      description='Implementation of Anchor method introduced in https://homes.cs.washington.edu/~marcotcr/aaai18.pdf',
      packages=['anchor2'],
      url='https://github.com/provectus/hydro-root-cause',
      author='Provectus',
      author_email='ygavrilin@provectus.com',
      license='BSD',
      install_requires=[
          'numpy',
          'pandas',
          'scikit-learn',
          'sklearn'
      ],
      include_package_data=True,
      zip_safe=False)
