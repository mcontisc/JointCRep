""" Functions for handling and visualizing the data. """

import networkx as nx
import numpy as np
import pandas as pd
import sktensor as skt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('white')


# Utils to handle the data

def import_data(dataset, undirected=False, ego='source', alter='alter', force_dense=True, noselfloop=True, verbose=True,
                binary=True):
    """
        Import data, i.e. the adjacency tensor, from a given folder.

        Return the NetworkX graph and its numpy adjacency tensor.

        Parameters
        ----------
        dataset : str
                  Path of the input file.
        undirected : bool
                     If set to True, the algorithm considers an undirected graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        force_dense : bool
                      If set to True, the algorithm is forced to consider a dense adjacency tensor.
        noselfloop : bool
                     If set to True, the algorithm removes the self-loops.
        verbose : bool
                  Flag to print details.
        binary : bool
                 Flag to force the matrix to be binary.

        Returns
        -------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
        B : ndarray
            Graph adjacency tensor.
        nodes : list
                List of nodes IDs.
    """

    # read adjacency file
    df_adj = pd.read_csv(dataset, sep='\s+')
    print('{0} shape: {1}'.format(dataset, df_adj.shape))

    # create the graph adding nodes and edges
    A = read_graph(df_adj=df_adj, ego=ego, alter=alter, undirected=undirected, noselfloop=noselfloop, verbose=verbose,
                   binary=binary)

    nodes = list(A[0].nodes)
    print('\nNumber of nodes =', len(nodes))
    print('Number of layers =', len(A))
    if verbose:
        print_graph_stat(A)

    # save the multilayer network in a tensor with all layers
    if force_dense:
        B, rw = build_B_from_A(A, nodes=nodes)
        B_T, data_T_vals = None, None
    else:
        B, B_T, data_T_vals, rw = build_sparse_B_from_A(A)

    return A, B, B_T, data_T_vals


def read_graph(df_adj, ego='source', alter='target', undirected=False, noselfloop=True, verbose=True, binary=True):
    """
        Create the graph by adding edges and nodes.

        Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.

        Parameters
        ----------
        df_adj : DataFrame
                 Pandas DataFrame object containing the edges of the graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        undirected : bool
                     If set to True, the algorithm considers an undirected graph.
        noselfloop : bool
                     If set to True, the algorithm removes the self-loops.
        verbose : bool
                  Flag to print details.
        binary : bool
                 If set to True, read the graph with binary edges.

        Returns
        -------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    # build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = list(set(egoID).union(set(alterID)))
    nodes.sort()

    L = df_adj.shape[1] - 2  # number of layers
    # build the multilayer NetworkX graph: create a list of graphs, as many graphs as there are layers
    if undirected:
        A = [nx.MultiGraph() for _ in range(L)]
    else:
        A = [nx.MultiDiGraph() for _ in range(L)]

    if verbose:
        print('Creating the network ...', end=' ')
    # set the same set of nodes and order over all layers
    for l in range(L):
        A[l].add_nodes_from(nodes)

    for index, row in df_adj.iterrows():
        v1 = row[ego]
        v2 = row[alter]
        for l in range(L):
            if row[l + 2] > 0:
                if binary:
                    if A[l].has_edge(v1, v2):
                        A[l][v1][v2][0]['weight'] = 1
                    else:
                        A[l].add_edge(v1, v2, weight=1)
                else:
                    if A[l].has_edge(v1, v2):
                        A[l][v1][v2][0]['weight'] += int(
                            row[l + 2])  # the edge already exists, no parallel edge created
                    else:
                        A[l].add_edge(v1, v2, weight=int(row[l + 2]))
    if verbose:
        print('done!')

    # remove self-loops
    if noselfloop:
        if verbose:
            print('Removing self loops')
        for l in range(L):
            A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

    return A


def print_graph_stat(G):
    """
        Print the statistics of the graph A.

        Parameters
        ----------
        G : list
            List of MultiDiGraph NetworkX objects.
    """

    L = len(G)
    N = G[0].number_of_nodes()

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
              f'{np.round(reciprocal_edges(G[l]), 3)}\n')


def build_B_from_A(A, nodes=None):
    """
        Create the numpy adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.
        nodes : list
                List of nodes IDs.

        Returns
        -------
        B : ndarray
            Graph adjacency tensor.
        rw : list
             List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
    """

    N = A[0].number_of_nodes()
    if nodes is None:
        nodes = list(A[0].nodes())
    B = np.empty(shape=[len(A), N, N])
    rw = []
    for l in range(len(A)):
        B[l, :, :] = nx.to_numpy_matrix(A[l], weight='weight', dtype=int, nodelist=nodes)
        rw.append(np.multiply(B[l], B[l].T).sum() / B[l].sum())

    return B, rw


def build_sparse_B_from_A(A):
    """
        Create the sptensor adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.

        Returns
        -------
        data : sptensor
               Graph adjacency tensor.
        data_T : sptensor
                 Graph adjacency tensor (transpose).
        v_T : ndarray
              Array with values of entries A[j, i] given non-zero entry (i, j).
        rw : list
             List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
    """

    N = A[0].number_of_nodes()
    L = len(A)
    rw = []

    d1 = np.array((), dtype='int64')
    d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    v, vT, v_T = np.array(()), np.array(()), np.array(())
    for l in range(L):
        b = nx.to_scipy_sparse_matrix(A[l])
        b_T = nx.to_scipy_sparse_matrix(A[l]).transpose()
        rw.append(np.sum(b.multiply(b_T)) / np.sum(b))
        nz = b.nonzero()
        nz_T = b_T.nonzero()
        d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
        d2 = np.hstack((d2, nz[0]))
        d2_T = np.hstack((d2_T, nz_T[0]))
        d3 = np.hstack((d3, nz[1]))
        d3_T = np.hstack((d3_T, nz_T[1]))
        v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
        vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
        v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))
    subs_ = (d1, d2, d3)
    subs_T_ = (d1, d2_T, d3_T)
    data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
    data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)

    return data, data_T, v_T, rw


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


def normalize_nonzero_membership(U):
    """
        Given a matrix, it returns the same matrix normalized by row.

        Parameters
        ----------
        U: ndarray
           Numpy Matrix.

        Returns
        -------
        The matrix normalized by row.
    """

    den1 = U.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return U / den1


def transpose_tensor(M):
    """
        Given M tensor, it returns its transpose: for each dimension a, compute the transpose ij->ji.

        Parameters
        ----------
        M : ndarray
            Tensor with the mean lambda for all entries.

        Returns
        -------
        Transpose version of M_aij, i.e. M_aji.
    """

    return np.einsum('aij->aji', M)


def expected_computation(B, U, V, W, eta):
    """
        Return the marginal and conditional expected value.

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        U : ndarray
            Out-going membership matrix.
        V : ndarray
            In-coming membership matrix.
        W : ndarray
            Affinity tensor.
        eta : float
              Pair interaction coefficient.

        Returns
        -------
        M_marginal : ndarray
                     Marginal expected values.
        M_conditional : ndarray
                        Conditional expected values.
    """

    lambda0_aij = lambda0_full(U, V, W)
    L = lambda0_aij.shape[0]

    Z = calculate_Z(lambda0_aij, eta)
    M_marginal = (lambda0_aij + eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z
    for l in np.arange(L):
        np.fill_diagonal(M_marginal[l], 0.)

    M_conditional = (eta ** transpose_tensor(B) * lambda0_aij) / (eta ** transpose_tensor(B) * lambda0_aij + 1)
    for l in np.arange(L):
        np.fill_diagonal(M_conditional[l], 0.)

    return M_marginal, M_conditional


def lambda0_full(u, v, w):
    """
        Compute the mean lambda0 for all entries.

        Parameters
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        M : ndarray
            Mean lambda0 for all entries.
    """

    if w.ndim == 2:
        M = np.einsum('ik,jk->ijk', u, v)
        M = np.einsum('ijk,ak->aij', M, w)
    else:
        M = np.einsum('ik,jq->ijkq', u, v)
        M = np.einsum('ijkq,akq->aij', M, w)

    return M


def calculate_Z(lambda0_aij, eta):
    """
        Compute the normalization constant of the Bivariate Bernoulli distribution.

        Returns
        -------
        Z : ndarray
            Normalization constant Z of the Bivariate Bernoulli distribution.
    """

    Z = lambda0_aij + transpose_tensor(lambda0_aij) + eta * np.einsum('aij,aji->aij', lambda0_aij, lambda0_aij) + 1
    for l in range(len(Z)):
        assert check_symmetric(Z[l])

    return Z


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
        Check if a matrix a is symmetric.
    """

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def compute_M_joint(U, V, W, eta):
    """
        Return the vectors of joint probabilities of every pair of edges.

        Parameters
        ----------
        U : ndarray
            Out-going membership matrix.
        V : ndarray
            In-coming membership matrix.
        W : ndarray
            Affinity tensor.
        eta : float
              Pair interaction coefficient.

        Returns
        -------
        [p00, p01, p10, p11] : list
                               List of ndarray with joint probabilities of having no edges, only one edge in one
                               direction and both edges for every pair of edges.
    """

    lambda0_aij = lambda0_full(U, V, W)

    Z = calculate_Z(lambda0_aij, eta)

    p00 = 1 / Z
    p10 = lambda0_aij / Z
    p01 = transpose_tensor(p10)
    p11 = (eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z

    return [p00, p01, p10, p11]


# Utils to visualize the data

def plot_hard_membership(graph, communities, pos, node_size, colors, edge_color):
    """
        Plot a graph with nodes colored by their hard memberships.
    """

    plt.figure(figsize=(10, 5))
    for i, k in enumerate(communities):
        plt.subplot(1, 2, i + 1)
        nx.draw_networkx(graph, pos, node_size=node_size, node_color=[colors[node] for node in communities[k]],
                         with_labels=False, width=0.5, edge_color=edge_color, arrows=True,
                         arrowsize=5, connectionstyle="arc3,rad=0.2")
        plt.title(k, fontsize=17)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def extract_bridge_properties(i, color, U, threshold=0.):
    groups = np.where(U[i] > threshold)[0]
    wedge_sizes = U[i][groups]
    wedge_colors = [color[c] for c in groups]
    return wedge_sizes, wedge_colors


def plot_soft_membership(graph, thetas, pos, node_size, colors, edge_color):
    """
        Plot a graph with nodes colored by their mixed (soft) memberships.
    """

    plt.figure(figsize=(10, 5))
    for j, k in enumerate(thetas):
        plt.subplot(1, 2, j + 1)
        ax = plt.gca()
        nx.draw_networkx_edges(graph, pos, width=0.5, edge_color=edge_color, arrows=True,
                               arrowsize=5, connectionstyle="arc3,rad=0.2", node_size=150, ax=ax)
        for i, n in enumerate(graph.nodes()):
            wedge_sizes, wedge_colors = extract_bridge_properties(i, colors, thetas[k])
            if len(wedge_sizes) > 0:
                _ = plt.pie(wedge_sizes, center=pos[n], colors=wedge_colors, radius=(node_size[i]) * 0.0005)
                ax.axis("equal")
        plt.title(k, fontsize=17)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_adjacency(Bd, M_marginal, M_conditional, nodes, cm='Blues'):
    """
        Plot the adjacency matrix and its reconstruction given by the marginal and the conditional expected values.
    """

    sns.set_style('ticks')
    plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    plt.subplot(gs[0, 0])
    im = plt.imshow(Bd[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.title('Data', fontsize=17)

    plt.subplot(gs[0, 1])
    plt.imshow(M_marginal[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.title(r'$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$', fontsize=17)

    plt.subplot(gs[0, 2])
    plt.imshow(M_conditional[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.title(r'$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$', fontsize=17)

    axes = plt.subplot(gs[0, 3])
    cbar = plt.colorbar(im, cax=axes)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.show()


def mapping(G, A):
    old = list(G.nodes)
    new = list(A.nodes)

    mapping = {}
    for x in old:
        mapping[x] = new[x]

    return nx.relabel_nodes(G, mapping)


def plot_graph(graph, M_marginal, M_conditional, pos, node_size, node_color, edge_color, threshold=0.2):
    """
        Plot a graph and its reconstruction given by the marginal and the conditional expected values.
    """

    plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)

    plt.subplot(gs[0, 0])
    edgewidth = [d['weight'] for (u, v, d) in graph.edges(data=True)]
    nx.draw_networkx(graph, pos, node_size=node_size, node_color=node_color, connectionstyle="arc3,rad=0.2",
                     with_labels=True, width=edgewidth, edge_color=edge_color, arrows=True, arrowsize=5,
                     font_size=15, font_color="black")
    plt.axis('off')
    plt.title('Data', fontsize=17)

    mask = M_marginal[0] < threshold
    M = M_marginal[0].copy()
    M[mask] = 0.
    G = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
    G = mapping(G, graph)
    edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]
    plt.subplot(gs[0, 1])
    nx.draw_networkx(G, pos, node_size=node_size, node_color=node_color, connectionstyle="arc3,rad=0.2",
                     with_labels=False, width=edgewidth, edge_color=edgewidth,
                     edge_cmap=plt.cm.Greys, edge_vmin=0, edge_vmax=1, arrows=True, arrowsize=5)
    plt.axis('off')
    plt.title(r'$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$', fontsize=17)

    mask = M_conditional[0] < threshold
    M = M_conditional[0].copy()
    M[mask] = 0.
    G = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
    G = mapping(G, graph)
    edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]

    plt.subplot(gs[0, 2])
    nx.draw_networkx(G, pos, node_size=node_size, node_color=node_color, connectionstyle="arc3,rad=0.2",
                     with_labels=False, width=edgewidth, edge_color=edgewidth,
                     edge_cmap=plt.cm.Greys, edge_vmin=0, edge_vmax=1, arrows=True, arrowsize=5)
    plt.axis('off')
    plt.title(r'$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$', fontsize=17)

    plt.tight_layout()
    plt.show()


def plot_precision_recall(conf_matrix, cm='Blues'):
    """
        Plot precision and recall of a given confusion matrix.
    """

    plt.figure(figsize=(10, 5))

    # normalized by row
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    plt.subplot(gs[0, 0])
    im = plt.imshow(conf_matrix / np.sum(conf_matrix, axis=1)[:, np.newaxis], cmap=cm, vmin=0, vmax=1)
    plt.xticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
    plt.yticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
    plt.ylabel('True', fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    plt.title('Precision', fontsize=17)

    # normalized by column
    plt.subplot(gs[0, 1])
    plt.imshow(conf_matrix / np.sum(conf_matrix, axis=0)[np.newaxis, :], cmap=cm, vmin=0, vmax=1)
    plt.xticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
    plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                    labelleft=False)
    plt.xlabel('Predicted', fontsize=15)
    plt.title('Recall', fontsize=17)

    axes = plt.subplot(gs[0, 2])
    plt.colorbar(im, cax=axes)

    plt.tight_layout()
    plt.show()


def plot_adjacency_samples(Bdata, Bsampled, cm='Blues'):
    """
        Plot the adjacency matrix and five sampled networks.
    """

    plt.figure(figsize=(30, 5))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1])
    plt.subplot(gs[0, 0])
    plt.imshow(Bdata[0], vmin=0, vmax=1, cmap=cm)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                    labelleft=False)
    plt.title('Data', fontsize=25)

    for i in range(5):
        plt.subplot(gs[0, i + 1])
        plt.imshow(Bsampled[i], vmin=0, vmax=1, cmap=cm)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                        left=False, labelleft=False)
        plt.title(f'Sample {i + 1}', fontsize=25)

    plt.tight_layout()
    plt.show()


