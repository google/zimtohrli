from setuptools import setup, find_packages, Extension

setup(
    name='pyohrli',
    version='0.2.1',
    author='Martin Bruse, Jyrki Alakuijala',
    author_email='zond@google.com, jyrki@google.com',
    description='Psychoacoustic perceptual metric that quantifies the human observable difference in two audio signals in the proximity of just-noticeable-differences',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    package_dir={'': 'cpp/zimt'},
    packages=find_packages(where='cpp/zimt'),
    py_modules=['pyohrli'],
    ext_modules=[
        Extension(
            name='_pyohrli',
            sources=['cpp/zimt/pyohrli.cc'],
            include_dirs=['cpp'],
        ),
    ],
    zip_safe=False,
    python_requires='>=3.8',
)
