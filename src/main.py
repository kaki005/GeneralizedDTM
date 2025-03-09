import logging
from math import log

import hydra
import omegaconf
import wandb
from utilpy import log_init

from configs import Config


@hydra.main(version_base=None, config_path="configs/", config_name="default")
def main(cfg: Config):
    log_init()
    logger = logging.getLogger("main")
    try:
        wandb.login()
        wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        loss = 0
        wandb.log({"loss": loss})
        wandb.finish()
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    main()
