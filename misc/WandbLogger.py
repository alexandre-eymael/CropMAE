import wandb

class WandbLogger:

    @staticmethod
    def make_image(*args, **kwargs):
        return wandb.Image(*args, **kwargs)

    @property
    def name(self):
        return wandb.run.name

    @property
    def id(self):
        return wandb.run.id

    def __init__(self, config, name=None, entity=None, project=None, mode=None):
        self.run = wandb.init(
            name=name,
            entity=entity,
            project=project,
            config=config,
            mode=mode,
            dir=".",
            reinit=True,
        )

    def log(self, metrics):
        wandb.log(metrics)

    def summary(self, key, value):
        wandb.run.summary[key] = value

    def get_summary(self, key):
        return wandb.run.summary[key]

    def finish(self):
        wandb.finish()

    def get_name(self):
        return wandb.run.name
    
    def get_config(self):
        return wandb.config