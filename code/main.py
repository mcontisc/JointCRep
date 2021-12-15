""" Code to perform the inference. """

import numpy as np
from argparse import ArgumentParser
import os
import yaml
import time
import sktensor as skt

import tools as tl
import JointCRep


def main():
    p = ArgumentParser()
    p.add_argument('-f', '--in_folder', type=str, default='../data/input/')  # path of the input folder
    p.add_argument('-j', '--adj_name', type=str, default='synthetic_data.dat')  # name of the adjacency tensor
    p.add_argument('-o', '--ego', type=str, default='source')  # name of the source of the edge
    p.add_argument('-r', '--alter', type=str, default='target')  # name of the target of the edge
    p.add_argument('-K', '--K', type=int, default=2)  # number of communities
    p.add_argument('-u', '--undirected', type=bool, default=False)  # flag to call the undirected network
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'], default='log')  # flag for convergence
    p.add_argument('-d', '--force_dense', type=bool, default=False)  # flag to force a dense transformation in input
    args = p.parse_args()

    tic = time.time()

    # save variables
    network = args.in_folder + args.adj_name  # network complete path
    K = args.K

    # setting to run the algorithm
    with open('setting_JointCRep.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # folder to store the inferred parameters
    if not os.path.exists(conf['out_folder']):
        os.makedirs(conf['out_folder'])
    conf['end_file'] += '_synthetic_data'

    '''
    Import data: removing self-loops and making binary
    '''
    A, B, B_T, data_T_vals = tl.import_data(network, ego=args.ego, alter=args.alter, undirected=args.undirected,
                                            force_dense=args.force_dense, noselfloop=True, verbose=True, binary=True)
    L = len(A)
    nodes = A[0].nodes()
    N = len(nodes)

    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)

    '''
    Run model
    '''
    mod_multicrep = JointCRep.joint_crep(N=N, L=L, K=K, undirected=args.undirected, **conf)
    _ = mod_multicrep.fit(data=B, data_T=B_T, data_T_vals=data_T_vals, flag_conv=args.flag_conv, nodes=nodes)

    toc = time.time()
    print(f'\n ---- Time elapsed: {np.round(toc-tic, 4)} seconds ----')


if __name__ == '__main__':
    main()
