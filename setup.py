#''' setup.py is needed, but only to make namespaces happen
from pathlib import Path

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().strip().split('\n')


#''' moved into function, can now be used other places
def version():
    for line in open('meta.yaml').readlines():
        index = line.find('set version')
        if index > -1:
            return line[index + 15:].replace('\" %}', '').strip()


setup(
    name='uxarray',
    version=version(),
    maintainer='UXARRAY',
    maintainer_email='',
    python_requires='>=3.7',
    install_requires=requirements,
    description=
    """Unstructured grid model reading and recognizing with xarray.""",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
    include_package_data=True,
    packages=find_packages(exclude=["docs", "test", "docs.*", "test.*"]),
    # namespace_packages=['UXARRAY'],
    url='https://github.com/UXARRAY/uxarray',
    project_urls={
        # 'Documentation': 'https://uxarray.readthedocs.io',
        'Source': 'https://github.com/UXARRAY/uxarray',
        'Tracker': 'https://github.com/UXARRAY/uxarray/issues',
    },
    zip_safe=False)
