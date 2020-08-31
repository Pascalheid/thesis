![Continuous Integration](https://github.com/Pascalheid/thesis/workflows/Continuous%20Integration/badge.svg)
[![Build Status](https://travis-ci.com/Pascalheid/thesis.svg?token=k9bHguJdFvokfiUmDB5q&branch=master)](https://travis-ci.com/Pascalheid/thesis)

## Replication Repo

In the folder [latex](https://github.com/Pascalheid/thesis/tree/master/latex) you can find my tex filex which compile the pdf which I handed in as my final thesis. 
It relies on the figures in the [figures](https://github.com/Pascalheid/thesis/tree/master/figures) folder. These as well as all other figures and tables of my thesis can be recreated by running the [replication notebook](https://github.com/Pascalheid/thesis/blob/master/replication.ipynb). This notebook relies on several python modules from which I import some functions. These modules can be found in the [python](https://github.com/Pascalheid/thesis/tree/master/python) folder. The [data](https://github.com/Pascalheid/thesis/tree/master/data) contains all the stored results from my simulations as well as the data from Iskhakov et al. (2016). 

I set up travis to execute my notebook as a replication check and the CI service of github to compile my thesis as pdf. My replication can also be accessed via mybinder. All necessary badges are included at the very top. 

In my thesis, I heavily rely on the [ruspy package](https://github.com/OpenSourceEconomics/ruspy) for which I made some contributions in the context of my thesis. These contributions can be seen in the Pull Requests [#42](https://github.com/OpenSourceEconomics/ruspy/pull/42) and [#46](https://github.com/OpenSourceEconomics/ruspy/pull/46). Additionally, I needed an implementation of the BHHH algorithm in Python which I had previously done in a project for Effective Programming Practices for Economists. During my thesis, I revised the BHHH and added a [PR to estimagic](https://github.com/OpenSourceEconomics/estimagic/pull/161). 

The current state is that I am still awaiting the PR for estimagic and the second PR for ruspy to be merged. Due to that, the travis build for the replication currently fails as both contributions are not available via conda, yet. In the meantime you can run my replication notebook only by, additionally to my provided environment, downloading the source code of estimagic and ruspy from my two PRs and installing them in editable mode. If you do so, my replication notebook runs through. 
