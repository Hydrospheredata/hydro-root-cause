from setuptools import setup

setup(name='rise',
      version='0.1',
      description='The fork of RISE (https://github.com/eclique/RISE)',
      py_modules=['rise'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=True)
