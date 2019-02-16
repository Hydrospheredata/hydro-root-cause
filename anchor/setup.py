from setuptools import setup

setup(name='anchor',
      version='0.0.1',
      description='The fork of acnhor_exp (https://github.com/marcotcr/anchor)',
      packages=['anchor'],
      url='http://github.com/marcotcr/anchor',
      author='Marco Tulio Ribeiro',
      author_email='marcotcr@gmail.com',
      license='BSD',
      install_requires=[
          'numpy',
          'pandas',
          'scikit-learn',
          'sklearn'
      ],
      include_package_data=True,
      zip_safe=False)
