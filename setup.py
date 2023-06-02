#############################################################################
#
#   Setup based on Labelme install script:
#   https://github.com/wkentaro/labelme
#
##############################################################################
import re
from setuptools import setup


def get_version():
    filename = "faceshine/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_install_requires():
    install_requires = [
        "numpy>=1.24.3",
        "torch>=2.0.0",
        "opencv-python>=4.7.0.72",
        "flask>=2.3.2",
        "flask_restful>=0.3.10"
    ]

    return install_requires


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()
    try:
        # when this package is being released
        import github2pypi

        return github2pypi.replace_url(
            slug="leonelhs/face-shine", content=long_description, branch="main"
        )
    except ImportError:
        # when this package is being installed
        return long_description


def main():
    setup(
        name='faceshine',
        version=get_version(),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        packages=['faceshine', 'faceshine.segmentation', 'faceshine.segmentation.base'],
        url='https://github.com/leonelhs/yapsy-gui',
        license='Apache',
        author='leonel hernandez',
        author_email='leonelhs@gmail.com',
        description='Photo image enhancer',
        install_requires=get_install_requires(),
        package_data={"faceshine": []},
        entry_points={"console_scripts": ["faceshine=faceshine.__main__:main"]},
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3 :: Only",
        ],
    )


if __name__ == "__main__":
    main()
