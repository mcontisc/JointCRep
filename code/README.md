# JointCRep: Python code
Copyright (c) 2021 [Martina Contisciani](https://www.is.mpg.de/person/mcontisciani), [Hadiseh Safdari](https://github.com/hds-safdari), and [Caterina De Bacco](http://cdebacco.com).

Implements the algorithm described in:

[1] Contisciani M., Safdari H., and De Bacco C. (2022). _Community detection and reciprocity in networks by jointly modeling pairs of edges_, Journal of Complex Networks 10, cnac034.

If you use this code please cite this [article](https://academic.oup.com/comnet/article/10/4/cnac034/6658441) (_published version_).     

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Files
- `JointCRep.py` : Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity. The latent variables are related to community memberships and a pair interaction value. This code is optimized to use sparse matrices.
- `main.py` : General version of the algorithm. It performs the inference in the given network by using the setting saved in `setting_JointCRep.yaml`, and infers the latent variables related to community and reciprocity.
-  `synthetic` : Class definition of the benchmark generative model used to generate synthetic data (*ReciprocityMMSBM_joints*). It creates a synthetic, directed, and binary network by a mixed-membership stochastic block-model with a reciprocity structure. Specifically, it models pairwise joint distributions with Bivariate Bernoulli distributions. An example on how to use this code is shown in the jupyter-notebook `generate_synthetic.ipynb`.
- `tools.py` : Contains non-class functions for handling and visualizing the data. 
- `test.py` : Code for testing the algorithm.
- `analysis_highschool.ipynb` : Example jupyter notebook to show the analysis of the highschool dataset.

## Usage
To test the program on the given synthetic example file, type

```bash
python main.py
```

It will use the example synthetic network contained in `../data/input`. The adjacency matrix *synthetic_data.dat* represents a directed and binary network with **N=250** nodes, **K=2** overlapping communities with an **assortative** structure, **⟨k⟩=20** average degree, and **$\eta$=50**.

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
