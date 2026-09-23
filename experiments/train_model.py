import logging
import os
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import yaml
from pathlib import Path
import pickle

from tsforge.models.transformer import Transformer
from tsforge.models.rnn import RNN
from tsforge.models.cnn import CNN
from tsforge.dataloaders.factory import DataLoaderFactory
from tsforge.common.train import train, eval_test
from tsforge.common._utils import set_determinism

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_TYPES = {
    "transformer": Transformer,
    "rnn": RNN,
    "cnn": CNN,
}

def get_model(name: str):
    if name not in MODEL_TYPES:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_TYPES.keys())}")
    return MODEL_TYPES[name]

def _load_dataset_file(path: str) -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, path)
    with open(full_path) as f:
        return yaml.safe_load(f)  # plain dict, not OmegaConf.create()

OmegaConf.register_new_resolver("load", _load_dataset_file)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    experiment_name = cfg.get("experiment_name", "default")
    output_dir = Path(f"outputs/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mcfg = OmegaConf.merge(
        OmegaConf.to_container(cfg.base,  resolve=True),
        OmegaConf.to_container(cfg.model, resolve=True),
    )
    mcfg = OmegaConf.create(mcfg)
    mcfg.horizon_override = getattr(cfg.dataset, "horizon_override", None)
    mcfg.context_len_override = getattr(cfg.dataset, "context_len_override", None)
    mcfg.n_channels = getattr(cfg.dataset, "n_channels", None)
    mcfg.checkpoint_dir = str(output_dir / "checkpoints")
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))

    if mcfg.horizon_override:
        mcfg.horizon = mcfg.horizon_override

    # ── Determinism ──────────────────────────────────────────────────────────
    # MUST stay above model construction. theta_0 is drawn from the global
    # torch stream, and torch's default seed is randomised per process, so a
    # model built before this call gets different weights on every run. The
    # torch.manual_seed inside fit() runs afterwards and cannot recover them.
    #
    # Safe to move earlier, never later. If you add anything above this line
    # that consumes RNG, this call still fixes everything below it.
    set_determinism(
        seed   = mcfg.seed,
        strict = getattr(mcfg, "deterministic", True),
    )

    factory = DataLoaderFactory(mcfg, dcfg)
    train_loader = factory.train_dataloader()
    val_loaders  = factory.val_dataloaders()
    model_cls    = get_model(mcfg.model_type)
    model        = model_cls(mcfg)

    train(
        model        = model,
        mcfg         = mcfg,
        train_loader = train_loader,
        val_loaders  = val_loaders,
        device       = torch.device(cfg.device),
        seed         = cfg.base.seed,
        resume       = cfg.get("resume", None),
    )

    results = eval_test(model, factory)
    with open(output_dir / "preds.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()