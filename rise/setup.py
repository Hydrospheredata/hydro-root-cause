from setuptools import setup

setup(name='root_cause_rise',
      version='0.1',
      description='The fork of RISE (https://github.com/eclique/RISE)',
      py_modules=['root_cause_rise'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=True)
