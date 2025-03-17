import logging
from math import log

import hydra
import omegaconf
import tensorflow as tf
import wandb
from gpflow.kernels.stationaries import Exponential
from tf_keras.mixed_precision import set_global_policy
from utilpy import log_init

from configs import Config
from models import GDTM, Corpus, preprocess_nips

tf.experimental.numpy.experimental_enable_numpy_behavior()


@hydra.main(version_base=None, config_path="configs/", config_name="default")
def main(cfg: Config):
    log_init()
    set_global_policy("float64")  # 全体のデフォルト dtype を float32 にする
    logger = logging.getLogger("main")
    try:
        # wandb.login()
        # wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        # wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        df, vocabulary, year = preprocess_nips()
        logger.info(f"{df.shape=}")
        logger.info(f"{vocabulary.shape=}")
        corpus = Corpus(df, vocabulary, year)
        kernel = Exponential()
        model = GDTM(kernel, 0.1, corpus, 3, 100, 0.1, 0.1, 0.1, True)
        model.inference_svi_gp(20, normalize_timestamps=True, test_schedule=10, epochs=100)
        # wandb.log({"loss": loss})
        # wandb.finish()
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    main()
