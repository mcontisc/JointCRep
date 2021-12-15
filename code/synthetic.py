""" Code to generate synthetic networks that emulates directed networks (possibly weighted)
with or without reciprocity. Self-loops are removed and only the largest connected component is considered. """

import os
import math
import warnings

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq

from abc import ABCMeta

DEFAULT_N = 1000
DEFAULT_L = 1
DEFAULT_K = 2
DEFAULT_ETA = 50
DEFAULT_ALPHA_HL = 6
DEFAULT_AVG_DEGREE = 15
DEFAULT_STRUCTURE = "assortative"

DEFAULT_PERC_OVERLAPPING = 0.2
DEFAULT_CORRELATION_U_V = 0.
DEFAULT_ALPHA = 0.1

DEFAULT_SEED = 10
DEFAULT_IS_SPARSE = True

DEFAULT_OUT_FOLDER = "data/input/synthetic/"

DEFAULT_SHOW_DETAILS = True
DEFAULT_SHOW_PLOTS = True
DEFAULT_OUTPUT_NET = True


def transpose_tensor(M):
    """
        Compute the transpose of a tensor with respect to the second and third dimensions.

        INPUT
        ----------
        M : ndarray
            Numpy tensor.

        OUTPUT
        -------
        Transpose of the matrix.
    """

    return np.einsum("aij->aji", M)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
        Check if a matrix a is symmetric in all layers.

        INPUT
        ----------
        a : ndarray
            Numpy matrix.

        OUTPUT
        -------
        symmetry : bool
                   Flag to assess if a matrix is symmetric in all layers.
    """
    symmetry = False
    for l in range(len(a)):
        symmetry = np.logical_and(np.allclose(a[l], a[l].T, rtol=rtol, atol=atol), symmetry)

    return symmetry


def normalize_nonzero_membership(u):
    """
        Given a matrix, it returns the same matrix normalized by row.

        INPUT
        ----------
        u: ndarray
           Numpy Matrix.

        OUTPUT
        -------
        The matrix normalized by row.
    """

    den1 = u.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1


def Exp_ija_matrix(u, v, w):
    """
        Compute the mean lambda0_ij for all entries.

        INPUT
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity matrix.

        OUTPUT
        -------
        M : ndarray
            Mean lambda0_ij for all entries.
    """

    if w.ndim == 2:
        M = np.einsum('ik,jk->ijk', u, v)
        M = np.einsum('ijk,ak->aij', M, w)
    else:
        M = np.einsum('ik,jq->ijkq', u, v)
        M = np.einsum('ijkq,akq->aij', M, w)

    return M


def build_edgelist(A, l):
    """
        Build the edgelist for a given layer, a in DataFrame format.

        INPUT
        ----------
        A : list
            List of scipy sparse matrices, one for each layer.
        l : int
            Layer number.

        OUTPUT
        -------
        df_res : DataFrame
                 Pandas DataFrame with edge information about a given layer.
    """

    A_coo = A.tocoo()
    data_dict = {'source': A_coo.row, 'target': A_coo.col, 'L'+str(l): A_coo.data}
    df_res = pd.DataFrame(data_dict)

    return df_res


def output_adjacency(A, out_folder, label):
    """
        Save the adjacency tensor to a file.
        Default format is space-separated .csv with L+2 columns: source_node target_node edge_l0 ... edge_lL

        INPUT
        ----------
        A : list
            List of scipy sparse matrices, one for each layer.
        out_folder : str
                     Path to store the adjacency tensor.
        label : str
                Label name to store the adjacency tensor.
    """

    outfile = label + '.dat'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    L = len(A)
    df = pd.DataFrame()
    for l in range(L):
        dfl = build_edgelist(A[l], l)
        df = df.append(dfl)
    df.to_csv(out_folder + outfile, index=False, sep=' ')
    print(f'Adjacency matrix saved in: {out_folder + outfile}')


def reciprocal_edges(G):
    """
        Compute the proportion of bi-directional edges, by considering the unordered pairs.

        Parameters
        ----------
        G: MultiDigraph
           MultiDiGraph NetworkX object.

        Returns
        -------
        reciprocity: float
                     Reciprocity value, intended as the proportion of bi-directional edges over the unordered pairs.
    """

    n_all_edge = G.number_of_edges()
    n_undirected = G.to_undirected().number_of_edges()  # unique pairs of edges, i.e. edges in the undirected graph
    n_overlap_edge = (n_all_edge - n_undirected)  # number of undirected edges reciprocated in the directed network

    if n_all_edge == 0:
        raise nx.NetworkXError("Not defined for empty graphs.")

    reciprocity = float(n_overlap_edge) / float(n_undirected)

    return reciprocity


def print_details(G):
    """
        Print the statistics of the graph A.

        Parameters
        ----------
        G : list
            List of MultiDiGraph NetworkX objects.
    """

    L = len(G)
    N = G[0].number_of_nodes()
    print('Number of nodes =', N)
    print('Number of layers =', L)
    print('Number of edges and average degree in each layer:')
    for l in range(L):
        E = G[l].number_of_edges()
        k = 2 * float(E) / float(N)
        print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')

        weights = [d['weight'] for u, v, d in list(G[l].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            M = np.sum([d['weight'] for u, v, d in list(G[l].edges(data=True))])
            kW = 2 * float(M) / float(N)
            print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')

        print(f'Sparsity [{l}] = {np.round(E / (N * N), 3)}')

        print(f'Reciprocity (networkX) = {np.round(nx.reciprocity(G[l]), 3)}')
        print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs) = '
              f'{np.round(reciprocal_edges(G[l]), 3)}')


def plot_A(A, cmap='PuBuGn'):
    """
        Plot the adjacency tensor produced by the generative algorithm.

        INPUT
        ----------
        A : list
            List of scipy sparse matrices, one for each layer.
        cmap : Matplotlib object
               Colormap used for the plot.
    """

    L = len(A)
    for l in range(L):
        Ad = A[l].todense()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(Ad, cmap=plt.get_cmap(cmap))
        ax.set_title(f'Adjacency matrix layer {l}', fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()


class BaseSyntheticNetwork(metaclass=ABCMeta):
    """
    A base abstract class for generation and management of synthetic networks.

    Suitable for representing any type of synthetic network.
    """

    def __init__(
        self,
        N: int = DEFAULT_N,
        L: int = DEFAULT_L,
        K: int = DEFAULT_K,
        seed: int = DEFAULT_SEED,
        out_folder: str = DEFAULT_OUT_FOLDER,
        output_net: bool = DEFAULT_OUTPUT_NET,
        show_details: bool = DEFAULT_SHOW_DETAILS,
        show_plots: bool = DEFAULT_SHOW_PLOTS,
        **kwargs
    ):

        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities

        # Set seed random number generator
        self.seed = seed
        self.prng = np.random.RandomState(self.seed)

        self.out_folder = out_folder
        self.output_net = output_net

        self.show_details = show_details
        self.show_plots = show_plots


class StandardMMSBM(BaseSyntheticNetwork):
    """
    Create a synthetic, directed, and weighted network (possibly multilayer)
    by a standard mixed-membership stochastic block-model
    - It models marginals (iid assumption) with Poisson distributions
    """

    def __init__(self, **kwargs):

        if "parameters" in kwargs:
            parameters = kwargs["parameters"]
        else:
            parameters = None

        self.init_mmsbm_params(**kwargs)

        self.build_Y(parameters=parameters)

        if self.output_net:
            self._output_parameters()
            output_adjacency(self.layer_graphs, self.out_folder, self.label)

        if self.show_details:
            print_details(self.G)
        if self.show_plots:
            plot_A(self.layer_graphs)
            if self.M is not None:
                self._plot_M()

    def init_mmsbm_params(self, **kwargs):
        """
        Check MMSBM-specific parameters
        """

        super().__init__(**kwargs)

        if "avg_degree" in kwargs:
            avg_degree = kwargs["avg_degree"]
            if avg_degree <= 0:  # (in = out) average degree
                err_msg = "The average degree has to be greater than 0.!"
                raise ValueError(err_msg)
        else:
            msg = f"avg_degree parameter was not set. Defaulting to avg_degree={DEFAULT_AVG_DEGREE}"
            warnings.warn(msg)
            avg_degree = DEFAULT_AVG_DEGREE
        self.avg_degree = avg_degree
        self.ExpEdges = int(self.avg_degree * self.N * 0.5)

        if "is_sparse" in kwargs:
            is_sparse = kwargs["is_sparse"]
        else:
            msg = f"is_sparse parameter was not set. Defaulting to is_sparse={DEFAULT_IS_SPARSE}"
            warnings.warn(msg)
            is_sparse = DEFAULT_IS_SPARSE
        self.is_sparse = is_sparse

        if "label" in kwargs:
            label = kwargs["label"]
        else:
            try:
                msg = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree_eta_seed"
                warnings.warn(msg)
                label = '_'.join([str(), str(self.N), str(self.L), str(self.K), str(self.avg_degree),
                                  str(self.eta), str(self.seed)])
            except:
                msg = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree_seed"
                warnings.warn(msg)
                label = '_'.join([str(), str(self.N), str(self.L), str(self.K), str(self.avg_degree), str(self.seed)])
        self.label = label

        """
        SETUP overlapping communities
        """
        if "perc_overlapping" in kwargs:
            perc_overlapping = kwargs["perc_overlapping"]
            if (perc_overlapping < 0) or (perc_overlapping > 1):  # fraction of nodes with mixed membership
                err_msg = "The percentage of overlapping nodes has to be in [0, 1]!"
                raise ValueError(err_msg)
        else:
            msg = f"perc_overlapping parameter was not set. Defaulting to perc_overlapping={DEFAULT_PERC_OVERLAPPING}"
            warnings.warn(msg)
            perc_overlapping = DEFAULT_PERC_OVERLAPPING
        self.perc_overlapping = perc_overlapping

        if self.perc_overlapping:
            # correlation between u and v synthetically generated
            if "correlation_u_v" in kwargs:
                correlation_u_v = kwargs["correlation_u_v"]
                if (correlation_u_v < 0) or (correlation_u_v > 1):
                    err_msg = "The correlation between u and v has to be in [0, 1]!"
                    raise ValueError(err_msg)
            else:
                msg = (f"correlation_u_v parameter for overlapping communities was not set. "
                       f"Defaulting to corr={DEFAULT_CORRELATION_U_V}")
                warnings.warn(msg)
                correlation_u_v = DEFAULT_CORRELATION_U_V
            self.correlation_u_v = correlation_u_v

            if "alpha" in kwargs:
                alpha = kwargs["alpha"]
            else:
                msg = f"alpha parameter of Dirichlet distribution was not set. Defaulting to alpha={[DEFAULT_ALPHA]*self.K}"
                warnings.warn(msg)
                alpha = [DEFAULT_ALPHA] * self.K
            if type(alpha) == float:
                if alpha <= 0:
                    err_msg = "Each entry of the Dirichlet parameter has to be positive!"
                    raise ValueError(err_msg)
                else:
                    alpha = [alpha] * self.K
            elif len(alpha) != self.K:
                err_msg = "The parameter alpha should be a list of length K."
                raise ValueError(err_msg)
            if not all(alpha):
                err_msg = "Each entry of the Dirichlet parameter has to be positive!"
                raise ValueError(err_msg)
            self.alpha = alpha

        """
        SETUP informed structure
        """
        if "structure" in kwargs:
            structure = kwargs["structure"]
        else:
            msg = f"structure parameter was not set. Defaulting to structure={[DEFAULT_STRUCTURE]*self.L}"
            warnings.warn(msg)
            structure = [DEFAULT_STRUCTURE] * self.L
        if type(structure) == str:
            if structure not in ["assortative", "disassortative"]:
                err_msg = "The available structures for the affinity tensor w are: assortative, disassortative!"
                raise ValueError(err_msg)
            else:
                structure = [structure] * self.L
        elif len(structure) != self.L:
            err_msg = ("The parameter structure should be a list of length L. "
                       "Each entry defines the structure of the corresponding layer!")
            raise ValueError(err_msg)
        for e in structure:
            if e not in ["assortative", "disassortative"]:
                err_msg = "The available structures for the affinity tensor w are: assortative, disassortative!"
                raise ValueError(err_msg)
        self.structure = structure

    def build_Y(self, parameters=None):
        """
        Generate network layers G using the latent variables,
        with the generative model A_ij ~ P(A_ij|u,v,w)
        """

        """
        Latent variables
        """
        if parameters is None:
            # generate latent variables
            self.u, self.v, self.w = self._generate_lv()
        else:
            # set latent variables
            self.u, self.v, self.w = parameters
            if self.u.shape != (self.N, self.K):
                raise ValueError('The shape of the parameter u has to be (N, K).')
            if self.v.shape != (self.N, self.K):
                raise ValueError('The shape of the parameter v has to be (N, K).')
            if self.w.shape != (self.L, self.K, self.K):
                raise ValueError('The shape of the parameter w has to be (L, K, K).')

        """
        Generate Y
        """
        self.M = Exp_ija_matrix(self.u, self.v, self.w)
        for l in range(self.L):
            np.fill_diagonal(self.M[l], 0)
        # sparsity parameter for Y
        if self.is_sparse:
            c = self.ExpEdges / self.M.sum()
            self.M *= c
            if parameters is None:
                self.w *= c

        Y = self.prng.poisson(self.M)

        """
        Create networkx DiGraph objects for each layer for easier manipulation
        """
        nodes_to_remove = []
        self.G = []
        self.layer_graphs = []
        for l in range(self.L):
            self.G.append(nx.from_numpy_matrix(Y[l], create_using=nx.DiGraph()))
            Gc = max(nx.weakly_connected_components(self.G[l]), key=len)
            nodes_to_remove.append(set(self.G[l].nodes()).difference(Gc))

        n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        for l in range(self.L):
            self.G[l].remove_nodes_from(list(n_to_remove))
            self.nodes = list(self.G[l].nodes())

            self.layer_graphs.append(nx.to_scipy_sparse_matrix(self.G[l], nodelist=self.nodes))

        self.u = self.u[self.nodes]
        self.v = self.v[self.nodes]
        self.N = len(self.nodes)

    def _apply_overlapping(self, u, v):
        """
            Introduce overlapping membership in the NxK membership vectors u and v, by using a Dirichlet distribution.

            INPUT, OUTPUT
            ----------
            u : Numpy array
                Matrix NxK of out-going membership vectors, positive element-wise.

            v : Numpy array
                Matrix NxK of in-coming membership vectors, positive element-wise.
        """

        overlapping = int(self.N * self.perc_overlapping)  # number of nodes belonging to more communities
        ind_over = self.prng.randint(len(u), size=overlapping)

        u[ind_over] = self.prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
        v[ind_over] = self.correlation_u_v * u[ind_over] + (1.0 - self.correlation_u_v) * \
                      self.prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
        if self.correlation_u_v == 1.0:
            assert np.allclose(u, v)
        if self.correlation_u_v > 0:
            v = normalize_nonzero_membership(v)

        return u, v
    
    def _sample_membership_vectors(self):
        """
            Compute the NxK membership vectors u and v without overlapping.

            OUTPUT
            ----------
            u : Numpy array
                Matrix NxK of out-going membership vectors, positive element-wise.

            v : Numpy array
                Matrix NxK of in-coming membership vectors, positive element-wise.
        """

        # Generate equal-size unmixed group membership
        size = int(self.N / self.K)
        u = np.zeros((self.N, self.K))
        v = np.zeros((self.N, self.K))
        for i in range(self.N):
            q = int(math.floor(float(i) / float(size)))
            if q == self.K:
                u[i:, self.K - 1] = 1.0
                v[i:, self.K - 1] = 1.0
            else:
                for j in range(q * size, q * size + size):
                    u[j, q] = 1.0
                    v[j, q] = 1.0

        return u, v

    def _compute_affinity_matrix(self, structure, a=0.1):
        """
            Compute the KxK affinity matrix w with probabilities between and within groups.

            INPUT
            ----------
            structure : list
                        List of structure of network layers.
            a : float
                Parameter for secondary probabilities.

            OUTPUT
            -------
            p : Numpy array
                Array with probabilities between and within groups. Element (k,h)
                gives the density of edges going from the nodes of group k to nodes of group h.
        """

        p1 = self.avg_degree * self.K / self.N

        if structure == "assortative":
            p = p1 * a * np.ones((self.K, self.K))  # secondary-probabilities
            np.fill_diagonal(p, p1 * np.ones(self.K))  # primary-probabilities

        elif structure == "disassortative":
            p = p1 * np.ones((self.K, self.K))  # primary-probabilities
            np.fill_diagonal(p, a * p1 * np.ones(self.K))  # secondary-probabilities

        return p

    def _generate_lv(self):
        """
            Generate latent variables for a MMSBM, assuming network layers are independent
            and communities are shared across layers.
        """

        # Generate u, v
        u, v = self._sample_membership_vectors()
        # Introduce the overlapping membership
        if self.perc_overlapping > 0:
            u, v = self._apply_overlapping(u, v)

        # Generate w
        w = np.zeros((self.L, self.K, self.K))
        for l in range(self.L):
            w[l, :, :] = self._compute_affinity_matrix(self.structure[l])

        return u, v, w

    def _output_parameters(self):
        """
            Output results in a compressed file.
        """

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        output_parameters = self.out_folder + 'gt_' + self.label
        try:
            np.savez_compressed(output_parameters + '.npz', u=self.u, v=self.v, w=self.w, eta=self.eta, nodes=self.nodes)
        except:
            np.savez_compressed(output_parameters + '.npz', u=self.u, v=self.v, w=self.w, nodes=self.nodes)
        print(f'Parameters saved in: {output_parameters}.npz')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')

    def _plot_M(self, cmap='PuBuGn'):
        """
            Plot the marginal means produced by the generative algorithm.

            INPUT
            ----------
            M : ndarray
                Mean lambda for all entries.
        """

        for l in range(self.L):
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(self.M[l], cmap=plt.get_cmap(cmap))
            ax.set_title(f'Marginal means matrix layer {l}', fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()


class ReciprocityMMSBM_joints(StandardMMSBM):
    """
    Proposed benchmark.
    Create a synthetic, directed, and binary network (possibly multilayer)
    by a mixed-membership stochastic block-model with a reciprocity structure
    - It models pairwise joint distributions with Bivariate Bernoulli distributions
    """

    def __init__(self, **kwargs):

        if "eta" in kwargs:
            eta = kwargs["eta"]
            if eta <= 0:  # pair interaction coefficient
                raise ValueError('The pair interaction coefficient eta has to greater than 0.!')
        else:
            msg = f"eta parameter was not set. Defaulting to eta={DEFAULT_ETA}"
            warnings.warn(msg)
            eta = DEFAULT_ETA
        self.eta = eta
        if "parameters" in kwargs:
            parameters = kwargs["parameters"]
        else:
            parameters = None

        super().init_mmsbm_params(**kwargs)

        self.build_Y(parameters=parameters)

        if self.output_net:
            super()._output_parameters()
            output_adjacency(self.layer_graphs, self.out_folder, self.label)

        if self.show_details:
            print_details(self.G)
        if self.show_plots:
            plot_A(self.layer_graphs)
            if self.M0 is not None:
                self._plot_M()

    def build_Y(self, parameters=None):
        """
        Generate network layers G using the latent variables,
        with the generative model (A_ij,A_ji) ~ P(A_ij, A_ji|u,v,w,eta)
        """

        """
        Latent variables
        """
        if parameters is None:
            # generate latent variables
            self.u, self.v, self.w = self._generate_lv()
        else:
            # set latent variables
            self.u, self.v, self.w = parameters
            if self.u.shape != (self.N, self.K):
                raise ValueError('The shape of the parameter u has to be (N, K).')
            if self.v.shape != (self.N, self.K):
                raise ValueError('The shape of the parameter v has to be (N, K).')
            if self.w.shape != (self.L, self.K, self.K):
                raise ValueError('The shape of the parameter w has to be (L, K, K).')

        """
        Generate Y
        """
        self.G = [nx.DiGraph() for _ in range(self.L)]
        self.layer_graphs = []

        nodes_to_remove = []
        for l in range(self.L):
            for i in range(self.N):
                self.G[l].add_node(i)

        # whose elements are lambda0_{ij}
        self.M0 = Exp_ija_matrix(self.u, self.v, self.w)
        for l in range(self.L):
            np.fill_diagonal(self.M0[l], 0)
            if self.is_sparse:
                # constant to enforce sparsity
                c = brentq(self._eq_c, 0.00001, 100., args=(self.ExpEdges, self.M0[l], self.eta))
                # print(f'Constant to enforce sparsity: {np.round(c, 3)}')
                self.M0[l] *= c
                if parameters is None:
                    self.w[l] *= c
        # compute the normalization constant
        self.Z = self._calculate_Z(self.M0, self.eta)

        for l in range(self.L):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    # [p00, p01, p10, p11]
                    probabilities = np.array([1., self.M0[l, j, i], self.M0[l, i, j],
                                              self.M0[l, i, j] * self.M0[l, j, i] * self.eta]) / self.Z[l, i, j]
                    cumulative = [1. / self.Z[l, i, j], np.sum(probabilities[:2]), np.sum(probabilities[:3]), 1.]
                    # print(f'({i}, {j}): {probabilities}')
                    r = self.prng.rand(1)[0]
                    if r <= probabilities[0]:
                        A_ij, A_ji = 0, 0
                    elif (r > probabilities[0]) and (r <= cumulative[1]):
                        A_ij, A_ji = 0, 1
                    elif (r > cumulative[1]) and (r <= cumulative[2]):
                        A_ij, A_ji = 1, 0
                    elif r > cumulative[2]:
                        A_ij, A_ji = 1, 1
                    if A_ij > 0:
                        self.G[l].add_edge(i, j, weight=1)  # binary
                    if A_ji > 0:
                        self.G[l].add_edge(j, i, weight=1)  # binary

            assert len(list(self.G[l].nodes())) == self.N

            # keep largest connected component
            Gc = max(nx.weakly_connected_components(self.G[l]), key=len)
            nodes_to_remove.append(set(self.G[l].nodes()).difference(Gc))

        n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        for l in range(self.L):
            self.G[l].remove_nodes_from(list(n_to_remove))
            self.nodes = list(self.G[l].nodes())

            self.layer_graphs.append(nx.to_scipy_sparse_matrix(self.G[l], nodelist=self.nodes))

        self.u = self.u[self.nodes]
        self.v = self.v[self.nodes]
        self.N = len(self.nodes)

    def _calculate_Z(self, lambda_aij, eta):
        """
            Compute the normalization constant of the Bivariate Bernoulli distribution.

            Parameters
            ----------
            lambda_aij : ndarray
                         Tensor with the mean lambda for all entries.
            eta : float
                  Reciprocity coefficient.

            Returns
            -------
            Z : ndarray
                Normalization constant Z of the Bivariate Bernoulli distribution.
        """

        Z = lambda_aij + transpose_tensor(lambda_aij) + eta * np.einsum('aij,aji->aij', lambda_aij, lambda_aij) + 1
        check_symmetric(Z)

        return Z

    def _eq_c(self, c, ExpM, M, eta):
        """
            Compute the function to set to zero to find the value of the sparsity parameter c.

            INPUT
            ----------
            c : float
                Sparsity parameter.
            ExpM : int
                   In-coming membership matrix.
            M : ndarray
                Mean lambda for all entries.
            eta : float
                  Reciprocity coefficient.

            OUTPUT
            -------
            Value of the function to set to zero to find the value of the sparsity parameter c.
        """

        LeftHandSide = (c * M + c * c * eta * M * M.T) / (c * M + c * M.T + c * c * eta * M * M.T + 1.)

        return np.sum(LeftHandSide) - ExpM

    def _plot_M(self, cmap='PuBuGn'):
        """
            Plot the marginal means produced by the generative algorithm.

            INPUT
            ----------
            cmap : Matplotlib object
                   Colormap used for the plot.
        """

        M = (self.M0 + self.eta * self.M0 * transpose_tensor(self.M0)) / self.Z
        for l in range(self.L):
            np.fill_diagonal(M[l], 0.)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(M[l], cmap=plt.get_cmap(cmap))
            ax.set_title(f'Marginal means matrix layer {l}', fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

