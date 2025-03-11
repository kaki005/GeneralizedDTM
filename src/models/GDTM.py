import logging
import time
from logging import Logger

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel
from pandas import Timestamp
from sklearn.feature_selection import SelectFdr
from sklearn.model_selection import train_test_split

from .Corpus import Corpus
from .datasets import get_neurips
from .utils import randn

Matrix = tf.Tensor
Vector = tf.Tensor


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
        self.training_doc_ids: tf.Tensor = tf.convert_to_tensor(training_doc_ids)
        """ids of the training documents"""
        self.validation_doc_ids: Vector = tf.convert_to_tensor(validation_doc_ids)
        """ ids of validation documents"""
        self.test_doc_ids: Vector = tf.convert_to_tensor(test_doc_ids)
        """ids of test documents"""
        if len(training_doc_ids) == 0:
            indices = np.arange(corpus.N)
            self.train_indices, self.test_indices = train_test_split(indices, test_size=0.2, random_state=42)
            self.logger.info(
                f"creating test set, using $test_doc_num for testing, {len(self.train_indices)} for training"
            )
        else:
            self.training_doc_ids = tf.convert_to_tensor(training_doc_ids)
        if len(test_doc_ids) != 0:
            self.test_doc_ids = tf.convert_to_tensor(test_doc_ids)

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
        self.likelihood_counts: list[int]  # list of #docs seen for likelihoods (ELBO estimates)
        self.test_counts: list[int]  # same but for test set likelihoods
        self.likelihoods: list[float]  # measured ELBO estimates
        self.test_perplexities: list[float]  # measured test set likelihoods
        self.learning_rates: list[float] = []  # computed learning rates
        self.word_observation_times: list[list[int]] = [[] for v in range(self.V)]
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
        self.times = tf.convert_to_tensor([t.to_julian_date() for t in self.timestamps])
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

    """
    The update step for local, i.e. document specific variational parameters. Computes the likelihood of the processed documents on the fly.
    """

    def e_step(self, data_idx: list[int]):
        e_step_ll = 0
        # 初期の words_seen を空集合で定義
        words_seen: set[int] = set()
        for doc_id in data_idx:
            doc = self.corpus.documents[doc_id]
            new_words = {word.id for word in doc.value.keys()}  # 各文書の単語 ID を取得
            words_seen = words_seen.union(new_words)  # 既存の集合と統合

        # reset sufficient statistics for the minibatch
        ɸ = [tf.zeros(self.T) for _ in range(self.K)]
        Ξ = [tf.zeros((self.T, self.V)) for _ in range(self.K)]
        timestamps_seen: set[int] = set()
        doc_topic_proportions = tf.zeros((len(data_idx), self.K))
        for i, doc_idx in enumerate(data_idx):  # 文書ごとに
            (doc_ll, λ) = self.doc_e_step(doc_idx, ɸ, Ξ, timestamps_seen)  # eステップ
            e_step_ll += doc_ll
            doc_topic_proportions[i, :] = λ  # 文書ごとのトピック比率
        return e_step_ll, ɸ, Ξ, words_seen, collect(timestamps_seen), doc_topic_proportions

    """
    Performs the e-step on individual documents.
    args:
        m: モデル
        doc_idx: 文書index
        ɸ: 時刻tのトピック割合 (トピック, 時間)
        Ξ: 時刻tに単語wがトピックkに割り当てられる確率 (トピック, 時間, 単語)
    return
        vi_ll: 対数尤度
        λ: 文書のトピック分布(Dirichlet)のパラメータ
    """

    def doc_e_step(self, doc_idx: int, ɸ: list[tf.Tensor], Ξ: list[tf.Tensor], timestamps_seen: set[int]):
        # extract word ids and frequencies from the document
        (doc_words, freqs) = self.corpus.get_words_and_counts(doc_idx)
        # extract timestamp of the documents
        t_doc = self.corpus.get_timestamp_index(doc_idx)
        timestamps_seen.add(t_doc)
        # do the actual inference
        (docφ, λ, vi_ll) = self.document_inference(t_doc, doc_words, freqs)
        # collect sufficient statistics
        for k in range(self.K):
            for i, w in enumerate(doc_words):
                ɸ[k][t_doc] += freqs[i] * docφ[i, k]
                Ξ[k][t_doc, w] += freqs[i] * docφ[i, k]
        return vi_ll, λ

    def document_inference(self, t_doc: int, doc_words: list[int], freqs: list[int]):
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
        test_schedule = 1
        cur_count = 0
        epochs = self.D // self.batch_size
        self.logger.info(f"doing {epochs} epochs with minibatch size {self.batch_size}")
        self.logger.info(f"parameter: Kernel: {self.krn} alpha: {self.alpha} K: {self.K}")
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
        self.Knn = (
            self.krn.K(self.times) + tf.eye(self.T) * self.s_x + jitter_full
        )  # compute training covariance matrix
        self.Knm = self.krn.K(self.times, self.inducing_points)  # compute cross covariance and some intermediate result
        self.KnmKmmInv = tf.matmul(self.Knm, self.KmmInv)
        self.K_tilde = self.Knn - tf.matmul(self.KnmKmmInv, self.Knm)
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
                self.eta_1[k][:, w] = -2 * tf.matmul(self.eta_2[k, w], self.mu[k][:, w])

        # assume s is identity matrix for init
        # compute the Λ diagonals and update 𝜁 parameter s
        init_val = tf.linalg.diag(tf.linalg.matmul(self.KnmKmmInv, self.KnmKmmInv.T))
        Λ_diags = tf.zeros((self.T, self.V))
        for k in range(self.K):
            for w in range(self.V):
                Λ_diags[:, w] = tf.identity(init_val)  # copy
            means = tf.matmul(self.KnmKmmInv, self.mu[k])
            self.svi_update_zeta(k, means, Λ_diags)
        # parameters to steer the step size in stochastic gradient updates
        a = 0.1
        b: int = 10
        gamma: float = 0.7
        # helpers for performing inference loop
        iter = 0
        svi_counter = 1
        perm = np.random.permutation(len(self.training_doc_ids))  # ランダムなインデックスの順列を取得
        mb_idx_rand = tf.gather(self.training_doc_ids, perm[: self.D])  # ランダムなnum_inducing個の要素を選択
        e_step_time_agg = 0.0
        m_step_time_agg = 0.0
        self.logger.info("done, starting inference")

        for e in range(epochs):
            start = time.perf_counter()
            iter += 1
            # determine the minibatch to operate on
            end_pos = svi_counter + self.batch_size - 1
            if end_pos > self.D:
                end_pos = self.D
            mb_idx = mb_idx_rand[svi_counter:end_pos]
            # keep track of documents seen
            cur_count += end_pos - svi_counter + 1
            # when reading the end of the training set, reshuffle the documents and start at beginning again
            svi_counter += self.batch_size
            if svi_counter > self.D:
                svi_counter = 1
                perm = np.random.permutation(self.training_doc_ids.size[0])  # ランダムなインデックスの順列を取得
                mb_idx_rand = self.training_doc_ids[perm[: self.D]]
                iter = 1

            # token count in the minibatch, gives a more realistic estimate of the fraction N/|S| for the gradient (instead of fractions of document counts)
            mb_wordcount = tf.reduce_sum(
                [tf.reduce_sum(list(doc["value"].values())) for doc in [self.corpus.documents[i] for i in mb_idx]]
            )  # TODO
            mult = all_tokens / mb_wordcount
            lr = a * (b + iter) ** (-gamma)  # compute the RM learning rate (using a common form)
            self.learning_rates.append(lr)
            ## ======================================
            ##do local udpate step (termed "e-step")
            ## ======================================
            (online_bound, ss_tk, ss_tkx, words_seen, t_mb, _) = self.e_step(mb_idx)
            online_bound *= mult  # multiply bound estimate by multiplier
            e_step_likelihoods.append(online_bound)  # estimate of document specific bound for whole corpus
            e_step_time = time.perf_counter() - start
            e_step_time_agg += e_step_time

            ## =====================================
            ##region (m-step, update global variables)
            ## =====================================
            start = time.perf_counter()
            m_step_bound = self.m_step(mult, ss_tk, ss_tkx, t_mb, lr, num_inducing)
            m_step_time = time.perf_counter() - start
            m_step_likelihoods.append(m_step_bound)
            online_bound += m_step_bound
            m_step_time_agg += m_step_time
            self.logger.info(f"epoch {e}, current elbo: {online_bound}, e-step: {e_step_time}, m-step: {m_step_time}")
            # endregion
            self.likelihoods.append(online_bound)
            self.likelihood_counts.append(cur_count)

            # test on unseen data according to schedule
            if self.test_doc_ids.size() > 0 and e % test_schedule == 0:
                test_ppx = self.test()
                self.test_counts.append(cur_count)
                self.test_perplexities.append(test_ppx)
                if test_schedule < 256:
                    test_schedule *= 2

        self.logger.info(
            f"time for GP SVI: {time.perf_counter() - start} seconds, average e-step time: {e_step_time_agg / epochs}, average m-step time: {m_step_time_agg / epochs}"
        )
        self.logger.info(f"e-step full: {e_step_time_agg}, m-step full: {m_step_time_agg}")

    def m_step(self, mult, ss_tk, ss_tkx, t_mb, lr: float, num_inducing: int) -> float:
        m_step_bound = 0.0
        KnmKmmInvEff = self.KnmKmmInv[t_mb, :]  # only look at words actually observed in the minibatch
        Λ_diags = tf.zeros((self.T, self.V))
        means = tf.zeros((self.T, self.V))
        word_bounds = tf.zeros(self.V)
        for k in range(self.K):
            ɸ_k = mult * ss_tk[k][t_mb]  # (T x 1)
            Ξ_k = mult * ss_tkx[k][t_mb, :]  # (T x V)
            for w in range(self.V):
                Λ_diags[t_mb, w] = tf.reshape(
                    tf.reduce_sum(tf.matmul(tf.matmul(KnmKmmInvEff, self.s[k, w]), KnmKmmInvEff), 2), [-1]
                )
            means[t_mb, :] = tf.matmul(KnmKmmInvEff, self.mu[k])
            B_tilde_k = ɸ_k * tf.exp(
                means[t_mb, :] + 0.5 * (Λ_diags[t_mb, :] + self.K_tilde_diag[t_mb]) - self.zeta[k, t_mb]
            )
            # euclidean gradient for mean
            dL_dm = KnmKmmInvEff.T * (Ξ_k + B_tilde_k * (means[t_mb, :] - 1))  # (12)
            self.eta_1[k] = (1 - lr) * self.eta_1[k] + lr * dL_dm
            for w in range(self.V):
                # euclidean gradient for variance
                dL_dS = -0.5 * (self.KmmInv + (KnmKmmInvEff * B_tilde_k[:, w]).T * KnmKmmInvEff)  # (12)
                # update step for second natural parameter
                self.eta_2[k, w] = (1 - lr) * self.eta_2[k, w] + lr * dL_dS
                # compute inverse and determinant using cholesky decomposition if possible
                eta_inv, det_eta_inv = tf.linalg.cholesky(self.eta_2[k, w] + self.jitter)
                # compute new covariance matrix and determinant (inducing points)
                self.s[k, w] = -0.5 * eta_inv
                det_s = (-0.5) ^ num_inducing * det_eta_inv
                # compute new mean (inducing points)
                self.mu[k][:, w] = tf.matmul(self.s[k, w], self.eta_1[k])
                Λ_diags[:, w] = tf.reshape(
                    tf.reduce_sum(tf.matmul(tf.matmul(self.KnmKmmInv, self.s[k, w]), self.KnmKmmInv), 2), [-1]
                )  # vectorize
                p_u = -0.5 * sum(tf.matmul(self.KmmInv, self.s[k, w]))
                q_u = -0.5 * tf.math.log(abs(det_s))
                word_bounds[w] = p_u - q_u
            # compute means for all timestamps (needed to recompute zeta)
            means = tf.matmul(self.KnmKmmInv, self.mu[k])
            m_step_bound += -0.5 * sum(tf.matmul(self.KmmInv, self.mu[k] * self.mu[k]))
            self.svi_update_zeta(k, means, Λ_diags)  # update zetas(auxiliary variable) again
            m_step_bound += sum(word_bounds)
        return m_step_bound

    """
    Do a heldout perplexity test. Learn topic proportions (given an optimized model) on the first half of a document, then compute the second half likelihood.
    Report the per-word predictive perplexity on the second half.
    """

    def test(self):
        test_ll = 0
        # set a seed, so that we are always looking at the same document parts
        tf.random.set_seed(12345)
        total_token_count = 0.0
        for doc_id in self.test_doc_ids:
            test_doc = self.corpus.documents[doc_id]
            t_doc = self.corpus.get_timestamp_index(test_doc.timestamp)
            (test_doc_words, test_freqs) = self.corpus.get_words_and_counts(test_doc.value)
            # expand tokens
            test_doc_tokens = []
            for i, w in enumerate(test_doc_words):
                for j in range(test_freqs[i]):
                    test_doc_tokens.append(w)
            n = len(test_doc_tokens)
            assert n == sum(test_freqs)
            # split document
            r = np.random.permutation(len(test_doc_tokens))[:n]
            n_2 = round(n / 2)
            w_1 = test_doc_tokens[r][1:n_2]
            tf_1 = tf.ones(n_2)
            w_2 = test_doc_tokens[r][n_2 + 1 :]
            tf_2 = tf.ones(len(w_2))
            # find optimal topic distribution
            (_, theta, _) = self.document_inference(t_doc, w_1, tf_1)
            # compute phis for second half
            means = tf.zeros((self.K, len(w_2)))
            for k in range(self.K):
                means[k, :] = tf.matmul(self.KnmKmmInv[t_doc, :].T, self.mu[k][:, w_2])

            phi = tf.zeros((len(w_2), self.K))
            dig_λ = tf.math.digamma(theta)
            dig_λ_sum = tf.math.digamma(sum(theta))
            for i in range(len(w_2)):
                log_phi_sum = 0
                for k in range(self.K):
                    phi[i, k] = (dig_λ[k] - dig_λ_sum) + (means[k, i] - self.zeta[k, t_doc])
                    log_phi_sum = phi[i, k] if k == 1 else tfp.math.log_add_exp(phi[i, k], log_phi_sum)
                # normalize and exp phi
                phi[i, :] = tf.exp(phi[i, :] - log_phi_sum) + 1e-100
            doc_ll = self.compute_document_likelihood(t_doc, w_2, tf_2, phi, theta, means)
            test_ll += doc_ll
            total_token_count += len(w_2)
        return -tf.exp(test_ll / total_token_count)

    def svi_update_zeta(self, k: int, means, Λ_diags):
        for t in range(self.T):
            self.zeta[k, t] = tf.math.reduce_logsumexp(means[t, :] + 0.5 * (Λ_diags[t, :] + self.K_tilde[t, t]))
