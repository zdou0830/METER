import os
import copy
import pytorch_lightning as pl
import os
os.environ["NCCL_DEBUG"] = "INFO"

from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.plugins.training_type import DDPPlugin
import torch.distributed as dist
class MyCluster(ClusterEnvironment):

    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        return True

    def master_address(self):
        return os.environ['MASTER_ADDR']

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    def world_size(self):
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])

    def global_rank(self) -> int:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])

    def local_rank(self) -> int:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    def node_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_NODE_RANK"])

    def set_global_rank(self, rank: int) -> None:
        pass

    def set_world_size(self, size: int) -> None:
        pass

class MyDDPPlugin(DDPPlugin):

    def init_ddp_connection(self, global_rank = None, world_size = None) -> None:
        master_uri = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        dist.init_process_group(
        backend=self.torch_distributed_backend,
        init_method=master_uri,
        world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
        rank=int(os.environ['OMPI_COMM_WORLD_RANK']),
        )

@ex.automain
def main(_config):
    os.environ["NCCL_DEBUG"] = "INFO"
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    # set environment variables for 'env://'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['NODE_RANK'] = str(os.environ["OMPI_COMM_WORLD_NODE_RANK"])

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = METERTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        plugins=[MyCluster(), MyDDPPlugin()],
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
