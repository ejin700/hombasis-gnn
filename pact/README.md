# PACT --  A Pattern Counting Toolkit
This is a toolkit for exact pattern counting, in particular, PACT currently supports exact subgraph counting for undirected and directed graphs.

Once everything has been set up (see below) we recommend to run
```sh
pipenv run jupyter notebook
```
from the main project directory and open the `Tutorial 1` notebook for an introduction on how to use Mobius.

## Installation
Necessary dependencies that are not installed automatically:
    - A working C compile chain
    - [go](https://go.dev/)
    - python 3.10 and pipenv
 
The rudimentary `setup.sh` script should set everything else up.

### Pynauty issues
The standard installation of `pynauty` doens't seem to work everywhere at the moment. If you are having problems with functionality that depends on nauty you can try the following in your work dir to reinstall pynauty in a more reliable way.
```sh
pipenv shell
pip uninstall pynauty
pip install --no-binary pynauty pynauty
```


