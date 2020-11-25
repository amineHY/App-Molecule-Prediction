from setuptools import setup

setup(
    name="predmol",
    version="1.0.0",
    license='MIT',
    author='Amine Hadj-Youcef',
    author_email='amine.hadjyoucef@gmail.com',
    description='This application uses machine learning to predict basic properties of a molecule...',
    packages=["app"],
    entry_points='''
        [console_scripts]
        predmol = app.cli:main
        '''
)
