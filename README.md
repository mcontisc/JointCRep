# JointCRep: community detection and reciprocity in networks by jointly modeling pairs of edges

Python implementation of JointCRep algorithm described in:

- [1] ADD LINK ARXIV

This is a new probabilistic generative model that takes into account community structure and reciprocity by specifying a closed-form joint distribution of a pair of network edges. To estimate the likelihood of network ties, we use a bivariate Bernoulli distribution where the log odds are linked to community memberships and pair interaction variables. 

This model aims to generalize the method [CRep](https://github.com/mcontisc/CRep) presented in [Safdari et al. (2021)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023209), which was of inspiration for the latent variables underlying the generative process.  

If you use this code please cite [1].   

Copyright (c) 2021 [Martina Contisciani](https://www.is.mpg.de/person/mcontisciani), [Hadiseh Safdari](https://github.com/hds-safdari), and [Caterina De Bacco](http://cdebacco.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## What's included
- `code` : Contains the Python implementation of JointCRep algorithm, the code for generating synthetic data with intrinsic community structure and reciprocity value, and two Jupyter-notebooks to show how to use the codes.
- `data/input` : Contains a synthetic example of directed network having an intrinsic community structure and reciprocity value, and the highschool dataset used in the manuscript. 
- `data/output` : Contains some results to test the code.

## Requirements
The project has been developed using Python 3.7 with the packages contained in *requirements.txt*. We suggest to create a conda environment with
`conda create --name JointCRep python=3.7.9 --no-default-packages`, activate it with `conda activate JointCRep`, and install all the dependencies by running (inside `JointCRep` directory):

```bash
pip install -r requirements.txt
```

## Test
You can run tests to reproduce results contained in `data/output` by running (inside `code` directory):  

```bash
python -m unittest test.py   
```

## Usage
To test the program on the given example file, type:  

```bash
cd code
python main.py
```

It will use the example synthetic network contained in `data/input`. The adjacency matrix *synthetic_data.dat* represents a directed and binary network generated with the jupyter-notebook *generate_synthetic.ipynb* by using the code *synthetic.py*.

### Parameters
- **-f** : Path of the input folder, *(default='../data/input/')*
- **-j** : Input file name of the adjacency matrix, *(default='synthetic_data.dat')*
- **-o** : Name of the source of the edge, *(default='source')*
- **-r** : Name of the target of the edge, *(default='target')*
- **-K** : Number of communities, *(default=2)*
- **-u** : Flag to call the undirected network, *(default=False)*
- **-F** : Flag to choose the convergence method, *(default='log')*
- **-d** : Flag to force a dense transformation of the adjacency matrix, *(default=False)*

You can find a list by running (inside `code` directory): 

```bash
python main.py --help
```

## Input format
The network should be stored in a *.dat* file. An example of rows is

`node1 node2 1` <br>
`node1 node3 1`

where the first and second columns are the _source_ and _target_ nodes of the edge, respectively, and the third column represents the presence of the edge. 


## Output
The algorithm returns a compressed file inside the `data/output` folder. To load and print the out-going membership matrix:

```bash
import numpy as np 
theta = np.load('theta_synthetic_data.npz')
print(theta['u'])
```

_theta_ contains the two NxK membership matrices **u** *('u')* and **v** *('v')*, the 1xKxK (or 1xK if assortative=True) affinity tensor **w** *('w')*, the pair interaction coefficient **$\eta$** *('eta')*, the total number of iterations *('max_it')*, the value of the maximum log-likelihood *('maxL')*, and the list of nodes of the network *('nodes')*.  
