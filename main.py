from torchlight import initialize_exp, set_seed,  get_dump_path
from runner import Runner
from data.dataloader import KVQA_Dataset
from configs import cfg
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    set_seed(cfg.random_seed)

    # Environment initialization...
    logger = initialize_exp(cfg)
    logger_path = get_dump_path(cfg)
    if not cfg.no_tensorboard:
        writer = SummaryWriter(
            log_dir=os.path.join(logger_path, 'tensorboard'))

    torch.cuda.set_device(cfg.gpu_id)

    # Run...
    runner = Runner(cfg, logger, writer)
    runner.run()
