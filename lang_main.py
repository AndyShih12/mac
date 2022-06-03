# IMPORTS
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from lang_runner import Runner
import wandb

@hydra.main(config_path="conf", config_name="config_lang")
def main(cfg: DictConfig):
    # absolute path
    cfg.data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    # relative to hydra path
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    config_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project="", config=config_dict)

    print(os.getcwd())
    print(OmegaConf.to_yaml(cfg))

    runner = Runner(cfg)

    if cfg.mode == 'train':
        runner.train()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
    
