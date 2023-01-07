from configs import cfg
from evaluator import Evaluator

if __name__ == "__main__":
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    evaluator = Evaluator(cfg)

    evaluator.eval()
