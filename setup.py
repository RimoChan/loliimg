import setuptools


setuptools.setup(
    name='loliimg',
    version='1.0.1',
    author='RimoChan',
    author_email='the@librian.net',
    description='loliimg',
    long_description=open('readme.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RimoChan/loliimg',
    packages=['loliimg'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'opencv-python>=4.5.1.48',
        'numpy>=1.19.5',
        'pyswarms>=1.3.0',
    ],
    python_requires='>=3.6',
)
