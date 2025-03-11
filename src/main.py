import logging
from math import log

import hydra
import omegaconf
import wandb
from configs import Config
from models import GDTM, Corpus, preprocess_nips
from utilpy import log_init


@hydra.main(version_base=None, config_path="configs/", config_name="default")
def main(cfg: Config):
    log_init()
    logger = logging.getLogger("main")
    try:
        # wandb.login()
        # wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        # wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        df, vocabulary, year = preprocess_nips()
        corpus = Corpus(df, vocabulary, year)
        model = GDTM(0.1, corpus, 3, 100, 0.1, 0.1, 0.1, True)
        model.inference_svi_gp(20)
        # wandb.log({"loss": loss})
        # wandb.finish()
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    main()
