from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        # Hint: self.k = k, self.sigma = sigma, etc.
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

#        raise NotImplementedError

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        # Hint: Start with computing centroid = X.mean(axis=0)

        # Add your code here

        #Step 1: Pre-compute
        n = X.shape[0]
        k_eff = min(self.k, n-1)

        c = X.mean(axis = 0)
        d = np.linalg.norm(X - c, axis = 1 ) 

        diff = X[:, None, :] - X[None, :, :]
        sq = (diff ** 2).sum(axis = 2)
        dist = np.sqrt(sq)

        if self.sigma is None:
            iu = np.triu_indices(n, k=1)
            sigma = 0.2 * np.median(dist[iu])

        else:
            sigma = self.sigma

        tau = 0.1 * sigma

        S = sq.copy()
        np.fill_diagonal(S, np.inf)

        nbr = np.argsort(S, axis=1, kind="stable")[:, :k_eff]
        nbr_sq = np.take_along_axis(sq, nbr, axis=1)

        w_ij = np.exp(-nbr_sq / sigma ** 2)


        d_nbr = d[nbr]
        downhill = d_nbr <= d[:, None] + tau

        #Step 2: Initialize Flux
        f = np.ones(n)

        #Step 3: Simulate Flow
        delta = np.zeros((n, k_eff))

        for t in range (self.T):
            if t == 0 or self.beta == 0:
                phi = np.ones(n)
            elif self.beta is None:
                phi = np.log(1 + f)
            else:
                phi = f ** self.beta

            w_tilde = w_ij * phi[nbr] * downhill

            den = w_tilde.sum(axis=1, keepdims=True)
            den_safe = np.where(den > 0, den, 1.0)
            frac = w_tilde / den_safe
            delta = self.alpha * frac * f[:, None]
            outgoing = delta.sum(axis=1)

            incoming = np.zeros(n)
            np.add.at(incoming, nbr, delta)
            f = f - outgoing + incoming

        #Step 4: Detect Sinks:
        flux = f

        out_final = delta.sum(axis=1)
        sinks = out_final < self.epsilon

        if not sinks.any():
            sinks[np.argmin(d)] = True

        #Step 5: Assign Clusters with Cycle Detection:
        masked = np.where(downhill, delta, -np.inf)
        pos = np.argmax(masked, axis=1)
        steepest = nbr[np.arange(n), pos]

        labels = np.full(n, -1)
        next_label = 0

        for s in np.where(sinks)[0]:
            labels[s] = next_label
            next_label += 1

        for i in range(n):
            if labels[i] != -1:
                continue

            path = []
            cur = i

            while labels[cur] == -1:
                path.append(cur)
                nxt = steepest[cur]

                if nxt in path:
                    labels[cur] = next_label
                    next_label += 1
                    break

                cur = nxt

            for p in path:
                labels[p] = labels[cur]


        #Step 6: Post-Processing: Small Cluster Merging:
        s_min = max(2, n//20)

        unique_labels, counts = np.unique(labels, return_counts=True)
        large = unique_labels[counts >= s_min]
        small = unique_labels[counts < s_min]

        if len(large) > 0:

            large_centroids = []

            for l in large:
                large_centroids.append(X[labels == l].mean(axis=0))
            large_centroids = np.array(large_centroids)


            for s in small:
                mu = X[labels == s].mean(axis=0)
                dists = np.linalg.norm(large_centroids - mu, axis=1)
                target = large[np.argmin(dists)]
                labels[labels == s] = target




        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
