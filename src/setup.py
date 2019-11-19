from os import system
from setuptools import setup, find_packages, Command

# visit: https://setuptools.readthedocs.io/en/latest/setuptools.html


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name='deus',
    version='1.0.0',

    author="Lucian Gomoescu, "
		   "Kennedy Putra Kusumo, "
           "Radoslav Paulen",
    author_email="kennedy.kusumo16@imperial.ac.uk, "
                 "gomoescu.lucian@gmail.com, "
                 "radoslav.paulen@stuba.sk",

    packages=find_packages(),

    cmdclass={'clean': CleanCommand}
)


