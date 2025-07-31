from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class PyohrliBuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == 'msvc':
            # MSVC is strict about designated initializers (and requires special
            # flag syntax anyway).
            cpp_standard_flag = '/std:c++20'
        else:
            cpp_standard_flag = '-std=c++17'

        for ext in self.extensions:
            ext.extra_compile_args.append(cpp_standard_flag)

        super().build_extensions()

setup(
    name='pyohrli',
    version='0.2.1',
    author='Martin Bruse, Jyrki Alakuijala',
    author_email='zond@google.com, jyrki@google.com',
    description='Psychoacoustic perceptual metric that quantifies the human observable difference in two audio signals in the proximity of just-noticeable-differences',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy>=1.20.0',
    ],
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
    cmdclass={'build_ext': PyohrliBuildExt},
    zip_safe=False,
    python_requires='>=3.10',
)
