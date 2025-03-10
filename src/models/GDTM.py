import logging
from logging import Logger

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel
from pandas import Timestamp
from sklearn.feature_selection import SelectFdr
from sklearn.model_selection import train_test_split

from .Corpus import Corpus
from .utils import randn

Matrix = tf.Tensor
Vector = list


class GDTM:
    def __init__(
        self,
        m_0: float,
        corpus: Corpus,
        num_topics: int,
        batch_size: int,
        s_0: float,
        s_x: float,
        alpha: float,
        visualize: bool,
        training_doc_ids: list[int] = [],
        validation_doc_ids: list[int] = [],
        test_doc_ids: list[int] = [],
        use_seeding: bool = False,
    ) -> None:
        self.logger: Logger = logging.getLogger(__class__)
        self.m_0: float = m_0
        """prior mean"""
        self.s_0: float = s_0
        """prior variance"""
        self.s_x: float = s_x
        """measurement variance"""
        self.invS: Matrix
        """inverse variance"""
        self.eta_1: Vector[Matrix]
        """natural parameters to the topic normals"""
        self.eta_2: Matrix
        """natural parameters to the topic normals"""
        self.alpha: float = alpha
        """alpha prior for dirichlets"""
        self.corpus: Corpus = corpus
        """data"""
        self.training_doc_ids: Vector = training_doc_ids
        """ids of the training documents"""
        self.validation_doc_ids: Vector = validation_doc_ids
        """ ids of validation documents"""
        self.test_doc_ids: Vector = test_doc_ids
        """ids of test documents"""
        if len(training_doc_ids) == 0:
            indices = np.arange(corpus.N)
            self.train_indices, self.test_indices = train_test_split(indices, test_size=0.2, random_state=42)
            self.logger.info(
                f"creating test set, using $test_doc_num for testing, {len(self.train_indices)} for training"
            )
        else:
            self.training_doc_ids = training_doc_ids
        if len(test_doc_ids) != 0:
            self.test_doc_ids = test_doc_ids

        self.timestamps: list[Timestamp] = corpus.get_unique_timestamps()
        """time stamp list (T)"""
        self.times: tf.Tensor = tf.zeros(1)
        """time points"""
        self.init_times()
        self.T: int = len(self.timestamps)
        """number of time points"""
        self.K: int = num_topics
        """ number of topics"""
        self.V: int = len(self.corpus.lexycon)
        """number of words"""
        self.batch_size: int = batch_size
        """size of minibatch"""
        self.D: int = len(self.training_doc_ids)
        """number of (training) documents"""
        self._lambda: Matrix
        """variational parameter to the dirichlets"""
        self.phi: Vector[Matrix]
        """variational parameter to the multinomials (selecting the source distribution)"""
        self.zeta: tf.Tensor
        """variational parameter for bounding the intractable expectation caused by softmax fct"""
        self.suff_stats_tk: Vector[Vector]
        """suffstats"""
        self.suff_stats_tkx: Vector[Matrix]
        """suffstats"""
        self.means: Vector[Matrix]
        """variational means"""
        self.s: Matrix
        """variational variance"""
        self.visualize: bool = visualize
        """switch on or off any visualization"""
        self.inducing_points: tf.Tensor  # inducing point locations for sparse GP
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
        self.mu: Vector[Matrix]  # inducing point values in sparse GP
        self.krn: Kernel  # kernel object for GP models
        self.likelihood_counts: Vector  # list of #docs seen for likelihoods (ELBO estimates)
        self.test_counts: Vector  # same but for test set likelihoods
        self.likelihoods: Vector  # measured ELBO estimates
        self.test_perplexities: Vector  # measured test set likelihoods
        self.learning_rates: Vector  # computed learning rates
        self.word_observation_times: Vector[Vector[int]] = [[] for v in range(self.V)]
        self.jitter: tf.Tensor  # diagonal
        self.use_seeding: bool = use_seeding

    def reset(self):
        self.likelihood_counts = []
        self.test_counts = []
        self.likelihoods = []
        self.test_perplexities = []
        self.learning_rates = []

        self.timestamps = self.corpus.get_unique_timestamps()
        self.init_times()
        self.T = len(self.times)
        # reset zeta parameter
        self.zeta = tf.zeros((self.K, self.T))

    def init_times(self):
        self.times = tf.Tensor([t.to_julian_date() for t in self.timestamps])
        self.times -= tf.reduce_min(self.times) - 1

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
                    log_phi_sum = phi[i, k] if k == 0 else tfp.math.log_add_exp(phi[i, k], log_phi_sum)
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

    def inference_svi_gp(self, num_inducing: int, rand_inducing: bool = False, normalize_timestamps: bool = False):
        e_step_likelihoods = []
        m_step_likelihoods = []
        full_elbo = []
        test_schedule = 1
        cur_count = 0
        epochs = self.D // self.batch_size
        self.logger.info("doing $epochs epochs with minibatch size $mini_batch_size")
        self.logger.info(f"parameter:\nKernel: {self.krn}\nalpha: {self.alpha}\nK: {self.K}")
        # timestamp normalization is disabled by default
        if normalize_timestamps:
            self.times -= tf.reduce_min(self.times) - 1
            self.times = self.times / tf.reduce_max(self.times)
        # ランダムな補助点選択はデフォルトでは無効で、タイムスタンプ上のドキュメントの分布はほぼ均一である。
        if num_inducing >= self.T:
            # if number of inducing points is larger than total timestamps, use all (degenerating into a full rank model)
            self.inducing_points = self.times
            num_inducing = self.T
        elif rand_inducing:  # randomly select induing points from time points
            perm = np.random.permutation(self.T)  # ランダムなインデックスの順列を取得
            self.inducing_points = tf.gather(self.times, perm[:num_inducing])  # ランダムなnum_inducing個の要素を選択
        else:
            self.inducing_points = tf.linspace(tf.reduce_min(self.times), tf.reduce_max(self.times), num_inducing)
        # 訓練セットの全トークンの数を計算し、尤度の平滑化に使用する。
        all_tokens = sum(sum(doc.value.values()) for doc in self.corpus.documents[self.training_doc_ids])

        # 補助点の共分散行列を計算
        self.Kmm = self.krn.K(self.inducing_points)
        self.jitter = tf.linalg.diag(tf.ones(num_inducing) * 1e-7)
        self.KmmInv = tf.linalg.inv(self.Kmm + self.jitter)
        jitter_full = tf.linalg.diag(tf.ones(self.T) * 1e-7)
        # compute training covariance matrix
        self.Knn = self.krn.K(self.times) + tf.eye(self.T) * self.s_x + jitter_full
        # compute cross covariance and some intermediate result
        self.Knm = self.krn.K(self.times, self.inducing_points)
        self.KnmKmmInv = self.Knm @ self.KmmInv
        self.K_tilde = self.Knn - self.KnmKmmInv * self.Knm
        self.K_tilde_diag = tf.linalg.diag(self.K_tilde)

        self.mu = tf.zeros(self.K)
        self.s = tf.zeros((self.K, self.V))
        self.eta_1 = tf.zeros(self.K)
        self.eta_2 = tf.zeros((self.K, self.V))
        L = tf.linalg.cholesky(self.Knn)  # (n, n)

        if self.use_seeding:
            for t in range(self.T):
                all_docs_t = self.corpus.get_datapoints_for_timestamp(self.timestamps[t])
                p = tf.zeros(self.K, dtype=tf.int64)
                if len(all_docs_t) < self.K:
                    for i in range(self.K):
                        p[i] = randn(len(all_docs_t) - 1)
                else:
                    perm = np.random.permutation(all_docs_t.shape[0])  # ランダムなインデックスの順列を取得
                    p = tf.gather(self.times, perm[: self.K])  # ランダムなnum_inducing個の要素を選択
                mean = tf.zeros(self.V)
                for k in range(self.K):
                    doc = self.corpus.documents[p[k]].value
                    (words, freqs) = self.corpus.get_words_and_counts(doc)
                    mean = tf.random.normal(shape=self.V) * self.s_0 + self.m_0
                    mean[words] = tf.math.log(freqs + randn(len(words) - 1))
                    # self.mu[k] = self.KnmKmmInv[t,:]' \ mean'

        # initialize variational covariances and natural parameters to topic distributions
        for k in range(self.K):
            if not self.use_seeding:
                self.mu[k] = tf.zeros((num_inducing, self.V))
            self.eta_1[k] = tf.zeros((num_inducing, self.V))
            for w in range(self.V):
                if not self.use_seeding:
                    self.mu[k][:, w] = tf.linalg.matmul(tf.linalg.inv(self.KnmKmmInv), L * randn(self.T))
                self.s[k, w] = tf.eye(num_inducing)
                self.eta_2[k, w] = -0.5 * tf.eye(num_inducing)
                self.eta_1[k][:, w] = -2 * self.eta_2[k, w] * self.mu[k][:, w]

            # assume s is identity matrix for init
            # compute the Λ diagonals and update 𝜁 parameters
            init_val = tf.linalg.diag(tf.linalg.matmul(self.KnmKmmInv, self.KnmKmmInv.T))
            Λ_diags = tf.zeros(self.T, self.V)
            for k in range(self.K):
                for w in range(self.V):
                    Λ_diags[:, w] = copy(init_val)
                means = tf.linalg.matmul(self.KnmKmmInv, self.mu[k])
                self.svi_update_zeta(k, means, Λ_diags)

            # parameters to steer the step size in stochastic gradient updates
            a = 0.1
            b = 10
            gamma = 0.7
            # helpers for performing inference loop
            iter = 0
            last_e_step_ll = -1e100
            svi_counter = 1
            perm = np.random.permutation(len(self.training_doc_ids))  # ランダムなインデックスの順列を取得
            mb_idx_rand = tf.gather(self.training_doc_ids, perm[: self.D])  # ランダムなnum_inducing個の要素を選択
            e_step_time_agg = 0.0
            m_step_time_agg = 0.0
            self.logger.info("done, starting inference")

    def svi_update_zeta(self, k: int, means, Λ_diags):
        for t in range(self.T):
            self.zeta[k, t] = log_sum(means[t, :] + 0.5 * (Λ_diags[t, :] + self.K_tilde[t, t]))
