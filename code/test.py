import unittest
import numpy as np
import JointCRep
import yaml
import tools as tl


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    in_folder = '../data/input/'
    adj = 'synthetic_data.dat'
    ego = 'source'
    alter = 'target'
    K = 2
    undirected = False
    flag_conv = 'log'
    force_dense = False

    with open('setting_JointCRep.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    out_folder = '../data/output/'
    conf['end_file'] += '_test'

    '''
    Import data: removing self-loops and making binary
    '''
    network = in_folder + adj  # network complete path
    A, B, B_T, data_T_vals = tl.import_data(network, ego=ego, alter=alter, undirected=undirected,
                                            force_dense=force_dense, noselfloop=True, verbose=True, binary=True)
    L = len(A)
    nodes = A[0].nodes()
    N = len(nodes)

    '''
    Run model
    '''
    mod_multicrep = JointCRep.joint_crep(N=N, L=L, K=K, undirected=undirected, **conf)
    _ = mod_multicrep.fit(data=B, data_T=B_T, data_T_vals=data_T_vals, flag_conv=flag_conv, nodes=nodes)

    # test case function to check the JointCRep.set_name function
    def test_import_data(self):
        print("Start import data test\n")
        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
            print('B has ', self.B.sum(), ' total weight.')
        else:
            self.assertTrue(self.B.vals.sum() > 0)
            print('B has ', self.B.vals.sum(), ' total weight.')

    # test case function to check the JointCRep.get_name function
    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        _ = self.mod_multicrep.fit(data=self.B, data_T=self.B_T, data_T_vals=self.data_T_vals,
                                   flag_conv=self.flag_conv, nodes=self.nodes)

        theta = np.load(self.mod_multicrep.out_folder+'theta'+self.mod_multicrep.end_file+'.npz')
        thetaGT = np.load(self.mod_multicrep.out_folder+'theta_synthetic_data.npz')

        self.assertTrue(np.array_equal(self.mod_multicrep.u_f, theta['u']))
        self.assertTrue(np.array_equal(self.mod_multicrep.v_f, theta['v']))
        self.assertTrue(np.array_equal(self.mod_multicrep.w_f, theta['w']))
        self.assertTrue(np.array_equal(self.mod_multicrep.eta_f, theta['eta']))

        self.assertTrue(np.array_equal(thetaGT['u'], theta['u']))
        self.assertTrue(np.array_equal(thetaGT['v'], theta['v']))
        self.assertTrue(np.array_equal(thetaGT['w'], theta['w']))
        self.assertTrue(np.array_equal(thetaGT['eta'], theta['eta']))


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
