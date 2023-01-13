import os
import yaml


class ConfigHandler():
    def __init__(self, datamodule, system, result_dir='./results/') -> None:
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        self.dm = datamodule
        self.sys = system

    def check_with_config(self):
        pass

    def record(self):
        with open(os.path.join(self.result_dir, "config.yaml"), 'w') as yf:
            yaml.dump({
                "data": self._extract_dataset_info(),
                "system": {
                    "_name": self.sys.__class__.__name__,
                    "model": {
                        "_name": self.sys.model.__class__.__name__,
                        "pretrained": self.sys.model.pretrained,
                    },
                    "loss": self.sys.loss.__class__.__name__,
                    "optimizer": {
                        "_name": self.sys.optimizer.__class__.__name__,
                        "lr": self.sys.optimizer.param_groups[0]['lr']
                    },
                },
            }, yf, default_flow_style=False)

    def _extract_dataset_info(self):
        info = {
            "datamodule": self.dm.__class__.__name__,
            "train": {"dataset": {}, "dataloader": {}},
            "val": {"dataset": {}, "dataloader": {}},
            "predict": {"dataset": {}, "dataloader": {}}
        }

        for phase in ["train", "val", "predict"]:
            if phase == "train" or phase == "val":
                self.dm.setup(stage='fit')
            elif phase == "predict":
                self.dm.setup(stage='predict')

            dataset = getattr(self.dm, f"{phase}_data")
            info[phase]['dataset']['_name'] = dataset.__class__.__name__
            try:
                info[phase]['dataset']['transform'] = dataset.transform.__name__
                for k, v in dataset.__dict__.items():
                    if callable(v) == False:
                        info[phase]['dataset'][k] = v
            except:
                pass

            dataloader = eval(f"self.dm.{phase}_dataloader")()
            for k, v in dataloader.__dict__.items():
                if callable(v) == False and k in [
                    "batch_size", "num_workers", "pin_memory"
                ]:
                    info[phase]['dataloader'][k] = v

        return info
