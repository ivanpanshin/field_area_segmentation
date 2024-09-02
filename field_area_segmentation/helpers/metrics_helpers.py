from hydra.utils import instantiate


def build_metrics(cfg):
    return [instantiate(v) for (k, v) in cfg.metric.items()]