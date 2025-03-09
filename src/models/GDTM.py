import tensorflow as tf
from gpflow.kernels import Kernel

from .utils import log_add

Matrix = tf.Tensor
Vector = list


class GDTM:
    def __init__(self) -> None:
        self.m_0: float  # prior mean
        self.s_0: float  # prior variance
        self.s_x: float  # measurement variance
        self.invS: Matrix  # inverse variance
        self.eta_1: Vector[Matrix]  # natural parameters to the topic normals
        self.eta_2: Matrix
        self.alpha: float  # alpha prior for dirichlets
        # self.corpus: CorpusUtils.Corpus  # the data
        self.training_doc_ids: Vector  # ids of the training documents
        self.validation_doc_ids: Vector  # ids of validation documents
        self.test_doc_ids: Vector  # ids of test documents
        self.T: int  # number of time points
        self.D: int  # number of (training) documents
        self.K: int  # number of topics
        self.V: int  # number of words
        self.batch_size: int  # size of minibatch

        self._lambda: Matrix  # variational parameter to the dirichlets
        self.phi: Vector[Matrix]  # variational parameter to the multinomials (selecting the source distribution)
        self.zeta: Matrix  # variational parameter for bounding the intractable expectation caused by softmax fct
        self.suff_stats_tk: Vector[Vector]  # suffstats
        self.suff_stats_tkx: Vector[Matrix]  # suffstats
        self.means: Vector[Matrix]  # variational means
        self.s: Matrix  # variational variance
        self.times: Vector  # time points
        self.visualize: bool  # switch on or off any visualization
        self.inducing_points: Vector  # inducing point locations for sparse GP
        self.Kmm: Matrix  # inducing point covariance for sparse GP
        self.KmmInv: Matrix  # inverse
        self.Knn: Matrix  # full rank covariance for GP models
        self.KnnInv: Matrix  # inverse
        self.Knm: Matrix  # cross covariance training points - inducing points for sparse GP
        self.KnmKmmInv: Matrix  # cross covariance x inverse inducing point covariance, to save computation
        self.K_tilde: Matrix  # low rank approximation of full rank covariance (sparse GP)
        self.K_tilde_diag: Vector  # diagonal of K_tilde
        self.S_diags: Vector[Matrix]  # variational covariance diagonals
        self.Λ_diags: Vector[Matrix]  # helper for storing Λ diagonals
        self.μ: Vector[Matrix]  # inducing point values in sparse GP
        self.krn: Kernel  # kernel object for GP models
        self.likelihood_counts: Vector  # list of #docs seen for likelihoods (ELBO estimates)
        self.test_counts: Vector  # same but for test set likelihoods
        self.likelihoods: Vector  # measured ELBO estimates
        self.test_perplexities: Vector  # measured test set likelihoods
        self.learning_rates: Vector  # computed learning rates
        self.word_observation_times: Vector[Vector[int]]
        self.jitter: Matrix  # diagonal
        self.use_seeding: bool

    def compute_document_likelihood(self, t_doc, doc_words, freqs, phi, lambd, means):
        likelihood = 0.0
        dig_lambda = tf.math.digamma(lambd)
        dig_lambda_sum = tf.math.digamma(tf.reduce_sum(lambd))
        for k in range(self.K):
            likelihood += tf.reduce_sum(
                freqs
                * phi[:, k]
                * (dig_lambda[k] - dig_lambda_sum + means[k, :] - self.zeta[k, t_doc] - tf.math.log(phi[:, k]))
            )
        likelihood += tf.reduce_sum(tf.math.lgamma(lambd)) - tf.math.lgamma(tf.reduce_sum(lambd))
        likelihood += tf.reduce_sum((self.alpha - lambd) * (dig_lambda - dig_lambda_sum))
        return likelihood

    def document_inference(self, t_doc, doc_words, freqs):
        last_vi_ll = -1e100
        vi_ll = 0
        converged = 1
        N_d = len(doc_words)  # get document data
        phi = tf.ones((len(doc_words), self.K)) / self.K  # Word-topic assignment probabilities
        lambd = tf.ones(self.K) * self.alpha + N_d / self.K  # Dirichlet distribution parameters
        dig_lambda = tf.math.digamma(lambd)
        dig_lambda_sum = tf.math.digamma(tf.reduce_sum(lambd))
        iter_count = 0
        means = tf.zeros((self.K, N_d))
        # Compute means for observed words in the document
        for k in range(self.K):
            means = tf.tensor_scatter_nd_update(
                means, [[k]], tf.linalg.matvec(self.KnmKmmInv[t_doc, :], self.mu[k][:, doc_words])
            )
        # Variational inference loop
        while converged > 1e-3:
            vi_ll = 0
            for i in range(N_d):
                log_phi_sum = 0
                for k in range(self.K):
                    phi = tf.tensor_scatter_nd_update(
                        phi, [[i, k]], [dig_lambda[k] - dig_lambda_sum + means[k, i] - self.zeta[k, t_doc]]
                    )
                    log_phi_sum = phi[i, k] if k == 0 else log_add(phi[i, k], log_phi_sum)
                # Normalize and exponentiate phi
                phi = tf.tensor_scatter_nd_update(phi, [[i]], [tf.exp(phi[i] - log_phi_sum) + 1e-100])
                # Update lambda
                lambd = self.alpha + tf.reduce_sum(freqs[:, None] * phi, axis=0)
                dig_lambda = tf.math.digamma(lambd)
                dig_lambda_sum = tf.math.digamma(tf.reduce_sum(lambd))
            # Compute document likelihood
            vi_ll = self.compute_document_likelihood(t_doc, doc_words, freqs, phi, lambd, means)
            # Check convergence
            converged = (last_vi_ll - vi_ll) / last_vi_ll
            last_vi_ll = vi_ll
            iter_count += 1
        return phi, lambd, vi_ll
