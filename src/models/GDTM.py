import logging
import time
from logging import Logger

import joblib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel
from pandas import Timestamp
from regex import B
from sklearn.feature_selection import SelectFdr
from sklearn.model_selection import train_test_split

from .Corpus import Corpus
from .datasets import get_neurips
from .utils import chol_inv, randn

Matrix = tf.Tensor
Vector = tf.Tensor


class GDTM:
    def __init__(
        self,
        kernel: Kernel,
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
        self.logger: Logger = logging.getLogger(f"{__class__}")
        self.m_0: float = m_0
        """prior mean"""
        self.s_0: float = s_0
        """prior variance"""
        self.s_x: float = s_x
        """measurement variance"""
        self.eta_1: list[tf.Variable]
        """natural parameters to the topic normals"""
        self.eta_2: tf.Variable
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
            self.training_doc_ids, self.test_doc_ids = train_test_split(indices, test_size=0.2, random_state=42)
            self.logger.info(
                f"creating test set, using $test_doc_num for testing, {len(self.training_doc_ids)} for training"
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
        self.D: int = self.training_doc_ids.shape[0]
        """number of (training) documents"""
        self.zeta: tf.Variable
        """variational parameter for bounding the intractable expectation caused by softmax fct"""
        self.s: tf.Variable
        """covariance matrix of inducing points"""
        self.visualize: bool = visualize
        """switch on or off any visualization"""
        self.inducing_points: tf.Tensor  # inducing point locations for sparse GP
        self.Kmm: Matrix
        """inducing point covariance for sparse GP"""
        self.KmmInv: Matrix  # inverse
        self.Knn: Matrix  # full rank covariance for GP models
        self.KnnInv: Matrix
        """inverse of Knn"""
        self.Knm: Matrix
        """cross covariance training points - inducing points for sparse GP"""
        self.KnmKmmInv: Matrix
        """cross covariance x inverse inducing point covariance, to save computation"""
        self.K_tilde: Matrix
        """low rank approximation of full rank covariance (sparse GP)"""
        self.K_tilde_diag: Matrix
        """diagonal of K_tilde"""
        self.mu: list[tf.Variable]
        """inducing point values in sparse GP"""
        self.krn: Kernel = kernel
        """kernel for GP models"""
        self.likelihood_counts: list[int]
        """list of #docs seen for likelihoods (ELBO estimates)"""
        self.test_counts: list[int]  # same but for test set likelihoods
        self.likelihoods: list[float]  # measured ELBO estimates
        self.test_perplexities: list[float]  # measured test set likelihoods
        self.learning_rates: list[float] = []  # computed learning rates
        self.word_observation_times: list[list[int]] = [[] for v in range(self.V)]
        self.jitter: tf.Tensor  # diagonal
        self.use_seeding: bool = use_seeding
        self.reset()
        self.logger.info(f"{self.D= }, {self.T=} ")

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
        self.zeta = tf.Variable(tf.zeros((self.K, self.T), dtype=tf.float64))

    def init_times(self):
        self.times = tf.expand_dims(
            tf.convert_to_tensor([t.to_julian_date() for t in self.timestamps], dtype=tf.float64), -1
        )
        self.times -= tf.reduce_min(self.times) - 1

    def compute_document_likelihood(self, t_doc, doc_words, freqs, phi, lambd, means):
        dig_lambda = tf.math.digamma(lambd)
        dig_lambda_sum = tf.math.digamma(tf.reduce_sum(lambd))
        a = (dig_lambda - dig_lambda_sum)[:, tf.newaxis] + means - self.zeta[:, t_doc, tf.newaxis] - tf.math.log(phi).T
        b = tf.matmul(phi, a)
        likelihood = tf.reduce_sum(tf.linalg.matvec(b, freqs, transpose_a=True))
        likelihood += tf.reduce_sum(tf.math.lgamma(lambd)) - tf.math.lgamma(tf.reduce_sum(lambd))
        likelihood += tf.reduce_sum((self.alpha - lambd) * (dig_lambda - dig_lambda_sum))
        return likelihood

    """
    The update step for local, i.e. document specific variational parameters. Computes the likelihood of the processed documents on the fly.
    """

    def e_step(self, data_idxes: tf.Tensor):
        e_step_ll = 0
        words_seen: set[int] = set()
        t_seen: set[int] = set()
        for doc_id in data_idxes:
            words, counts = self.corpus.get_words_and_counts(doc_id)
            words_seen = words_seen.union(words)  # 既存の集合と統合
            t_doc = self.corpus.get_timestamp_index(doc_id.numpy().item())
            t_seen.add(t_doc)

        # reset sufficient statistics for the minibatch
        ɸ = tf.zeros((self.T, self.K), dtype=tf.float64)
        Ξ = tf.zeros((self.T, self.V, self.K), dtype=tf.float64)
        doc_topic_proportions = []
        returns = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.doc_e_step)(doc_idx.numpy().item()) for doc_idx in data_idxes
        )
        for ret in returns:
            (vi_ll, λ, ɸt, Ξt) = ret
            e_step_ll += vi_ll
            doc_topic_proportions.append(λ)
            ɸ += ɸt
            Ξ += Ξt
        # for i, doc_idx in enumerate(data_idxes):  # 文書ごとに
        #     (vi_ll, λ, ɸ, Ξ) = self.doc_e_step(doc_idx.numpy().item(), ɸ, Ξ, t_seen)  # eステップ
        #     e_step_ll += vi_ll
        #     doc_topic_proportions.append(λ)  # 文書ごとのトピック比率
        return e_step_ll, ɸ, Ξ, words_seen, t_seen, doc_topic_proportions

    """
    Performs the e-step on individual documents.
    args:
        m: モデル
        doc_idx: 文書index
        ɸ: 時刻tのトピック割合 (時間,トピック)
        Ξ: 時刻tに単語wがトピックkに割り当てられる確率 (時間, 単語,トピック)
    return
        vi_ll: 対数尤度
        λ: 文書のトピック分布(Dirichlet)のパラメータ
    """

    # def doc_e_step(self, doc_idx: int, ɸ: tf.Tensor, Ξ: tf.Tensor):
    def doc_e_step(self, doc_idx: int):
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        ɸ = tf.zeros((self.T, self.K), dtype=tf.float64)
        Ξ = tf.zeros((self.T, self.V, self.K), dtype=tf.float64)
        # extract word ids and frequencies from the document
        (doc_words, freqs) = self.corpus.get_words_and_counts(doc_idx)
        t_doc = self.corpus.get_timestamp_index(doc_idx)  # timestamp of the documents
        # do the actual inference (docφ:(単語, トピック))
        (docφ, λ, vi_ll) = self.document_inference(t_doc, doc_words, freqs)
        # collect sufficient statistics
        add = tf.linalg.matvec(docφ, freqs, transpose_a=True)  # (トピック)
        ɸ = tf.tensor_scatter_nd_add(ɸ, tf.constant([[t_doc]]), [add])
        # adds = tf.linalg.matvec(docφ, freqs, transpose_a=True)
        indices = tf.constant([[t_doc, w] for w in doc_words])
        Ξ = tf.tensor_scatter_nd_add(Ξ, indices, tf.transpose(tf.transpose(docφ) * freqs))
        # for i, w in enumerate(doc_words):
        #     add = tf.gather(docφ, tf.constant([i]), axis=0) * freqs[i]
        #     Ξ = tf.tensor_scatter_nd_add(Ξ, tf.constant([[t_doc, w]]), add)
        return vi_ll, λ, ɸ, Ξ

    def document_inference(self, t_doc: int, doc_words: list[int], freqs: tf.Tensor):
        last_vi_ll = -1e100
        vi_ll = 0
        converged = 1
        N_d = len(doc_words)  # get document data
        phi = tf.ones((N_d, self.K), dtype=tf.float64) / self.K  # Word-topic assignment probabilities
        λ = tf.ones(self.K, dtype=tf.float64) * self.alpha + N_d / self.K  # Dirichlet distribution parameters
        dig_lambda = tf.math.digamma(λ)
        dig_lambda_sum = tf.math.digamma(tf.reduce_sum(λ))
        iter_count = 0
        means = tf.zeros((self.K, N_d), dtype=tf.float64)
        # Compute means for observed words in the document
        for k in range(self.K):
            a = tf.gather(self.KnmKmmInv, t_doc, axis=0)
            b = tf.gather(self.mu[k], doc_words, axis=1)
            update = tf.expand_dims(tf.linalg.matvec(b, a, transpose_a=True), axis=0)
            means = tf.tensor_scatter_nd_update(means, tf.constant([[k]]), update)
        # Variational inference loop
        while converged > 1e-3:
            vi_ll = 0
            for i in range(N_d):
                updates = (
                    dig_lambda
                    - dig_lambda_sum
                    + tf.squeeze(tf.gather(means, tf.constant([i]), axis=1))
                    - tf.squeeze(tf.gather(self.zeta, tf.constant([t_doc]), axis=1))
                )

                phi = tf.tensor_scatter_nd_update(phi, tf.constant([[i]]), [updates])
                log_phi_sum = tf.math.reduce_logsumexp(phi)
                # Normalize and exponentiate phi
                phi = tf.tensor_scatter_nd_update(phi, [[i]], [tf.exp(phi[i] - log_phi_sum) + 1e-100])
                # Update lambda
                lambd = self.alpha + tf.linalg.matvec(phi, freqs, transpose_a=True)
                dig_lambda = tf.math.digamma(lambd)
                dig_lambda_sum = tf.math.digamma(tf.reduce_sum(lambd))
            # Compute document likelihood
            vi_ll = self.compute_document_likelihood(t_doc, doc_words, freqs, phi, lambd, means)
            # Check convergence
            converged = (last_vi_ll - vi_ll) / last_vi_ll
            last_vi_ll = vi_ll
            iter_count += 1
        return phi, lambd, vi_ll

    def inference_svi_gp(
        self, num_inducing: int, rand_inducing: bool = False, normalize_timestamps: bool = False, test_schedule=1
    ):
        e_step_likelihoods = []
        m_step_likelihoods = []
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
        self.inducing_points = tf.expand_dims(self.inducing_points, -1)
        # 訓練セットの全トークンの数を計算し、尤度の平滑化に使用する。
        all_tokens = np.sum([np.sum(self.corpus.get_words_and_counts(doc_idx)[1]) for doc_idx in self.training_doc_ids])

        # 補助点の共分散行列を計算
        self.Kmm = self.krn.K(self.inducing_points)
        self.jitter = tf.linalg.diag(tf.ones(num_inducing, dtype=tf.float64) * 1e-7)
        self.KmmInv = tf.linalg.inv(self.Kmm + self.jitter)
        jitter_full = tf.linalg.diag(tf.ones(self.T, dtype=tf.float64) * 1e-7)
        self.Knn = (
            self.krn.K(self.times) + tf.eye(self.T, dtype=tf.float64) * self.s_x + jitter_full
        )  # compute training covariance matrix
        self.Knm = self.krn.K(self.times, self.inducing_points)  # compute cross covariance and some intermediate result
        self.KnmKmmInv = tf.matmul(self.Knm, self.KmmInv)
        self.K_tilde = self.Knn - tf.matmul(self.KnmKmmInv, self.Knm, transpose_b=True)
        self.mu = []
        self.s = tf.Variable(tf.zeros((self.K, self.V, num_inducing, num_inducing), dtype=tf.float64))
        self.eta_1 = []
        self.eta_2 = tf.Variable(tf.zeros((self.K, self.V, num_inducing, num_inducing), dtype=tf.float64))
        K_tilde_diag = tf.linalg.diag_part(self.K_tilde)
        L = tf.linalg.cholesky(self.Knn)  # (n, n)
        # muの初期化
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
                    (words, freqs) = self.corpus.get_words_and_counts(p[k])
                    mean = tf.random.normal(shape=self.V) * self.s_0 + self.m_0
                    mean[words] = tf.math.log(freqs + randn(len(words) - 1))
                    self.mu.append(tf.Variable(tf.linalg.solve(self.KnmKmmInv[t, :].T, mean.T), dtype=tf.float64))
        else:
            for k in range(self.K):
                self.mu.append(tf.Variable(tf.zeros((num_inducing, self.V), dtype=tf.float64)))
                for w in range(self.V):
                    z = tf.random.normal((self.T, 1), dtype=tf.float64)
                    Lz = tf.linalg.matmul(L, z)
                    self.mu[k][:, w].assign(tf.squeeze(tf.linalg.lstsq(self.KnmKmmInv, Lz)))
        # initialize variational covariances and natural parameters to topic distributions
        for k in range(self.K):
            self.eta_1.append(tf.Variable(tf.zeros((num_inducing, self.V), dtype=tf.float64)))
            for w in range(self.V):
                self.s[k, w].assign(tf.eye(num_inducing, dtype=tf.float64))
                self.eta_2[k, w].assign(-0.5 * tf.eye(num_inducing, dtype=tf.float64))
                self.eta_1[k][:, w].assign(-2 * tf.linalg.matvec(self.eta_2[k, w], self.mu[k][:, w]))

        # assume s is identity matrix for init
        # compute the Λ diagonals and update 𝜁 parameter s
        init_val = tf.linalg.diag_part(tf.linalg.matmul(self.KnmKmmInv, self.KnmKmmInv, transpose_b=True))
        for k in range(self.K):
            Λ_diags = tf.tile(tf.expand_dims(init_val, [-1]), tf.constant([1, self.V], dtype=tf.int32))  # copy ()
            means = tf.matmul(self.KnmKmmInv, self.mu[k])
            self.svi_update_zeta(k, means, Λ_diags)
        # parameters to steer the step size in stochastic gradient updates
        a = 0.1
        b: int = 10
        gamma: float = 0.7  # helpers for performing inference loop
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
                [tf.reduce_sum(self.corpus.get_words_and_counts(doc_idx)[1]) for doc_idx in mb_idx]
            )  # TODO
            mult = all_tokens / mb_wordcount
            lr = a * (b + iter) ** (-gamma)  # compute the RM learning rate (using a common form)
            self.learning_rates.append(lr)
            ## ======================================
            ##do local udpate step (termed "e-step")
            ## ======================================
            online_bound, ɸ, Ξ, words_seen, t_seen, doc_topic_proportions = self.e_step(mb_idx)
            online_bound *= mult  # multiply bound estimate by multiplier
            e_step_likelihoods.append(online_bound)  # estimate of document specific bound for whole corpus
            e_step_time = time.perf_counter() - start
            e_step_time_agg += e_step_time
            ## =====================================
            ##region (m-step, update global variables)
            ## =====================================
            start = time.perf_counter()
            m_step_bound = self.m_step(mult, ɸ, Ξ, list(t_seen), lr, num_inducing, K_tilde_diag)
            m_step_time = time.perf_counter() - start
            m_step_likelihoods.append(m_step_bound)
            online_bound += m_step_bound
            m_step_time_agg += m_step_time
            self.logger.info(f"epoch {e}, current elbo: {online_bound}, e-step: {e_step_time}, m-step: {m_step_time}")
            # endregion
            self.likelihoods.append(online_bound)
            self.likelihood_counts.append(cur_count)
            # test on unseen data according to schedule
            if self.test_doc_ids.shape[0] > 0 and e % test_schedule == 0:
                test_ppx = self.test()
                self.test_counts.append(cur_count)
                self.test_perplexities.append(test_ppx)
                if test_schedule < 256:
                    test_schedule *= 2

        self.logger.info(
            f"average e-step time: {e_step_time_agg / epochs}, average m-step time: {m_step_time_agg / epochs}"
        )
        self.logger.info(f"e-step full: {e_step_time_agg}, m-step full: {m_step_time_agg}")

    def m_step(
        self, mult, ɸ: tf.Tensor, Ξ: tf.Tensor, t_mb: list[int], lr: float, num_inducing: int, K_tilde_diag: tf.Tensor
    ) -> float:
        m_step_bound = 0.0
        KnmKmmInvEff = tf.gather(self.KnmKmmInv, t_mb, axis=0)  # only look at words actually observed in the minibatch
        Λ_diags = tf.zeros((self.T, self.V), dtype=tf.float64)
        means = tf.zeros((self.T, self.V), dtype=tf.float64)
        for k in range(self.K):
            word_bounds = []
            ɸ_k = mult * tf.gather(ɸ, t_mb)[:, k]  # (T ✖︎ 1)
            Ξ_k = mult * tf.gather(Ξ, t_mb)[:, :, k]  # (T ✖︎ V)
            update = tf.transpose(
                [tf.reduce_sum(tf.matmul(KnmKmmInvEff, self.s[k, w]) * KnmKmmInvEff, axis=1) for w in range(self.V)]
            )
            Λ_diags = tf.tensor_scatter_nd_update(Λ_diags, [[t] for t in t_mb], update)
            means = tf.tensor_scatter_nd_update(means, [[t] for t in t_mb], tf.matmul(KnmKmmInvEff, self.mu[k]))
            temp = means + 0.5 * (K_tilde_diag[:, tf.newaxis] + Λ_diags) - self.zeta[k, :, tf.newaxis]
            B_tilde_k = ɸ_k[:, tf.newaxis] * tf.exp(tf.gather(temp, t_mb))
            # euclidean gradient for mean
            dL_dm = tf.matmul(KnmKmmInvEff, (Ξ_k + B_tilde_k * tf.gather(means, t_mb) - 1), transpose_a=True)  # (12)
            self.eta_1[k] = (1 - lr) * self.eta_1[k] + lr * dL_dm  # natural gradient update
            for w in range(self.V):
                # euclidean gradient for variance
                dL_dS = -0.5 * (
                    self.KmmInv + tf.matmul(KnmKmmInvEff * B_tilde_k[:, w, tf.newaxis], KnmKmmInvEff, transpose_a=True)
                )  # (12)
                # update step for second natural parameter
                self.eta_2[k, w].assign((1 - lr) * self.eta_2[k, w] + lr * dL_dS)
                # compute inverse and determinant using cholesky decomposition if possible
                cov_inv, det_cov_inv = chol_inv(-(self.eta_2[k, w] + self.jitter))
                eta2_inv = -cov_inv
                det_eta2_inv = -det_cov_inv
                # compute new covariance matrix and determinant (inducing points)
                self.s[k, w].assign(0.5 * eta2_inv)
                det_s = (-0.5) ** num_inducing * det_eta2_inv
                # compute new mean (inducing points)
                self.mu[k][:, w].assign(tf.linalg.matvec(self.s[k, w], self.eta_1[k][:, w]))
                p_u = -0.5 * tf.reduce_sum(tf.matmul(self.KmmInv, self.s[k, w]))
                q_u = -0.5 * tf.math.log(abs(det_s))
                word_bounds.append(p_u - q_u)
            # compute means for all timestamps (needed to recompute zeta)
            means = tf.matmul(self.KnmKmmInv, self.mu[k])
            m_step_bound += -0.5 * tf.reduce_sum(tf.matmul(self.KmmInv, self.mu[k]) * self.mu[k])
            a = tf.matmul(self.KnmKmmInv, self.s[k]) * self.KnmKmmInv
            Λ_diags = tf.transpose(tf.reduce_sum(a, 2))
            self.svi_update_zeta(k, means, Λ_diags)  # update zetas(auxiliary variable) again
            m_step_bound += tf.reduce_sum(word_bounds)
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
            t_doc = self.corpus.get_timestamp_index(doc_id)
            (test_doc_words, test_freqs) = self.corpus.get_words_and_counts(doc_id)
            test_doc_tokens = []
            for i, w in enumerate(test_doc_words):
                count = tf.cast(test_freqs[i], tf.int32).numpy().item()
                test_doc_tokens += [w] * count
            n = len(test_doc_tokens)
            assert n == tf.cast(tf.reduce_sum(test_freqs), tf.int32).numpy().item()
            test_doc_tokens = np.array(test_doc_tokens)
            # split document
            r = np.random.permutation(n).tolist()
            n_2 = round(n / 2)
            w_1 = tf.convert_to_tensor(test_doc_tokens[r][:n_2])
            tf_1 = tf.ones(n_2, dtype=tf.float64)
            w_2 = tf.convert_to_tensor(test_doc_tokens[r][n_2:])
            tf_2 = tf.ones(w_2.shape[0], dtype=tf.float64)
            # find optimal topic distribution
            (_, theta, _) = self.document_inference(t_doc, w_1, tf_1)
            # compute phis for second half
            means = tf.zeros((self.K, w_2.shape[0]), dtype=tf.float64)
            for k in range(self.K):
                means = tf.tensor_scatter_nd_update(
                    means,
                    [[k]],
                    [
                        tf.linalg.matvec(
                            tf.gather(self.mu[k], w_2, axis=1), self.KnmKmmInv[t_doc, :].T, transpose_a=True
                        )
                    ],
                )

            phi = tf.zeros((w_2.shape[0], self.K), dtype=tf.float64)
            dig_λ = tf.math.digamma(theta)
            dig_λ_sum = tf.math.digamma(tf.reduce_sum(theta))
            phi = (dig_λ - dig_λ_sum)[:, tf.newaxis] + means - self.zeta[:, t_doc, tf.newaxis]
            log_phi_sum = tf.math.reduce_logsumexp(phi, axis=0)
            phi = tf.exp(phi - log_phi_sum) + 1e-100  # normalize and exp phi
            doc_ll = self.compute_document_likelihood(t_doc, w_2, tf_2, phi.T, theta, means)
            test_ll += doc_ll
            total_token_count += w_2.shape[0]
        return -tf.exp(test_ll / total_token_count)

    def svi_update_zeta(self, k: int, means, Λ_diags):
        for t in range(self.T):
            self.zeta[k, t].assign(
                tf.math.reduce_logsumexp(means[t, :] + 0.5 * (Λ_diags[t, :] + self.K_tilde[t, t]))
            )  # A.1
