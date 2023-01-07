import os
from pprint import pprint
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchlight import initialize_exp, set_seed, snapshot, get_dump_path, show_params
from data.dataloader import KVQA_Dataset
from model.answer_mlp import MLP
from model import SimpleClassifier
from model import fusion_net
from utils import freeze_layer, Metrics, cosine_sim, instance_bce_with_logits


class Runner:
    def __init__(self, args, logger, tb_writer=None):
        # prepare for: data , model, loss fuction, optimizer

        self.args = args
        self.logger = logger
        self.tb_writer = tb_writer
        self.log_dir = get_dump_path(args)
        self.model_dir = os.path.join(self.log_dir, "model")
        self.fusion_model = None
        self.answer_net = None

        # data load
        self.dataset = KVQA_Dataset(args)
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.TRAIN.batch_size,
            shuffle=True,  # only shuffle the data in training
            pin_memory=True,
            num_workers=args.TRAIN.data_workers,
        )

        # get the fusion_model and answer_net
        self._model_choice(args)
        # self._load_model(self.fusion_model, "fusion")
        # self._load_model(self.answer_net, "embedding")

        # self.best_answer_net = self.answer_net.clone()

        # optimizer
        params_for_optimization = list(self.fusion_model.parameters()) + list(
            self.answer_net.parameters()
        )
        self.optimizer = optim.Adam(
            [p for p in params_for_optimization if p.requires_grad], lr=args.TRAIN.lr
        )

        # scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=args.TRAIN.CosineAnnealing_Tmax, T_mult=args.TRAIN.CosineAnnealing_Tmax, eta_min=1e-7)

        # loss fuction
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.loss_func = nn.MSELoss().cuda()

        # Recorder
        self.max_acc = [0, 0, 0, 0]
        self.max_zsl_acc = [0, 0, 0, 0]
        self.best_epoch = 0
        self.correspond_loss = 1e20

        self.early_stop = 0

        print("fusion_model:")
        pprint(self.fusion_model)
        print("Answer Model:")
        pprint(self.answer_net)

    def run(self):
        # warm up(ref: ramen)
        # self.gradual_warmup_steps = [
        #     i * self.args.TRAIN.lr for i in torch.linspace(0.5, 2.0, 7)]
        # self.lr_decay_epochs = range(14, 47, self.args.TRAIN.lr_decay_step)
        for epoch in range(self.args.TRAIN.epochs):
            self.early_stop += 1
            if self.args.patience < self.early_stop:
                # early stop
                break
            # warm up
            # if epoch < len(self.gradual_warmup_steps):
            #     self.optimizer.param_groups[0]['lr'] = self.gradual_warmup_steps[epoch]
            # elif epoch in self.lr_decay_epochs:
            #     self.optimizer.param_groups[0]['lr'] *= self.args.TRAIN.lr_decay_rate

            self.train_metrics = Metrics()
            self.val_metrics = Metrics()
            self.zsl_metrics = Metrics()

            if not self.args.now_test:
                ######## begin training!! #######
                self.train(epoch)
                #################################
                lr = self.optimizer.param_groups[0]['lr']
                # recode:
                self.logger.info(
                    f'Train Epoch {epoch}: LOSS={self.train_metrics.total_loss: .6f}, lr={lr: .6f}, acc1={self.train_metrics.acc_1: .2f},acc3={self.train_metrics.acc_3: .2f},acc10={self.train_metrics.acc_10: .2f}')

            if epoch % 1 == 0 and epoch > 0:
                ######## begin evaling!! #######
                # self.eval(epoch)
                #################################
                # logger.info(
                #     '#################################################################################################################')
                # logger.info(
                #     f'Test Epoch {epoch}: LOSS={self.val_metrics.total_loss: .5f}, acc1={self.val_metrics.acc_1: .2f}, acc3={self.val_metrics.acc_3: .2f}, acc10={self.val_metrics.acc_10: .2f}')
                # if args.ZSL and not self.args.fact_map and not args.relation_map:
                #     logger.info(
                #         f'Zsl Epoch {epoch}: LOSS={self.zsl_metrics.total_loss: .5f}, acc1={self.zsl_metrics.acc_1: .2f}, acc3={self.zsl_metrics.acc_3: .2f}, acc10={self.zsl_metrics.acc_10: .2f}')
                # logger.info(
                #     '#################################################################################################################')

                # add 0.1 accuracy punishment, avoid for too much attention on hit@10 acc
                # if self.val_metrics.total_loss < (self.correspond_loss - 1) or self.val_metrics.acc_all > (self.max_acc[3] + 0.2):
                # reset early_stop and updata
                if self.train_metrics.acc_1 > self.max_acc[0]:
                    self.early_stop = 0
                    self.best_epoch = epoch
                    self.correspond_loss = self.val_metrics.total_loss
                    self._update_best_result(self.max_acc, self.val_metrics)

                    # self.best_fusion_model = copy.deepcopy(self.fusion_model)
                    # self.best_answer_net = copy.deepcopy(self.answer_net)
                    # save the model
                    if not self.args.now_test and self.args.save_model:
                        self.fusion_model_path = self._save_model(
                            self.fusion_model, "fusion")
                        self.answer_net_path = self._save_model(
                            self.answer_net, "embedding")

                if not self.args.no_tensorboard and not self.args.now_test:
                    self.tb_writer.add_scalar(
                        'loss', self.val_metrics.total_loss, epoch)
                    self.tb_writer.add_scalar(
                        'acc1', self.val_metrics.acc_1, epoch)
                    self.tb_writer.add_scalar(
                        'acc3', self.val_metrics.acc_3, epoch)
                    self.tb_writer.add_scalar(
                        'acc10', self.val_metrics.acc_10, epoch)

            self.scheduler.step()

    def train(self, epoch):
        self.fusion_model.train()
        self.answer_net.train()
        prefix = "train"
        tq = tqdm(self.loader, desc='{} E{:03d}'.format(
            prefix, epoch), ncols=0)

        if self.args.space_name == 'answer':
            KGE_space = torch.from_numpy(
                self.dataset.KGE_answer).cuda()
        else:
            KGE_space = self.dataset.rel_embeddings.cuda()

        for poi, question_features, answers, answers_onehot, q_len in tq:
            poi = Variable(poi.float()).cuda()
            question_features = Variable(question_features).cuda()
            answers = Variable(answers).cuda()
            answers_onehot = Variable(answers_onehot).cuda()
            q_len = Variable(q_len).cuda()

            fusion_embedding = self.fusion_model(
                poi, question_features, q_len)

            if self.args.answer_model == 'CLS':
                # TODO: Normalization?
                predicts = self.answer_net(fusion_embedding)
                loss = instance_bce_with_logits(predicts, answers_onehot)
                arg_predicts = torch.argmax(
                    predicts, dim=1).detach().cpu().numpy()
                arg_answer = torch.argmax(
                    answers_onehot, dim=1).detach().cpu().numpy()
                a = 1
            else:
                # Mapping-based methods
                a = 1
                loss = self.loss_func(fusion_embedding, answers)
                fusion_embedding_ = fusion_embedding.unsqueeze(dim=0)
                KGE_space_ = KGE_space.unsqueeze(dim=0)

                predicts = torch.cdist(fusion_embedding_, KGE_space_, p=2)
                predicts = predicts.squeeze()
                # predicts = predicts.to(torch.float64)
                predicts = -self.log_softmax(predicts).to(torch.float64)
                a = 1

                # answer_embedding = self.answer_net(KGE_space)

                # notice the temperature (correspoding to specific score)
                # predicts = cosine_sim(
                #     fusion_embedding, answer_embedding) / self.args.loss_temperature
                # predicts = predicts.to(torch.float64)
                # nll = -self.log_softmax(predicts).to(torch.float64)
                # loss = nll.mean()
                # multi answers case?
                # loss = (nll * answers / answers.sum(1,
                #         keepdim=True)).sum(dim=1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_metrics.update_per_batch(
                loss, answers_onehot.data, predicts.data)
        self.train_metrics.update_per_epoch()

    def _model_choice(self, args):
        assert args.fusion_model in ['SAN', 'MLP']
        # models api
        self.fusion_model = getattr(fusion_net, args.fusion_model)(
            args, self.dataset.num_tokens).cuda()
        # freeze word embedding
        if args.freeze_w2v and args.fusion_model != 'MLP':
            freeze_layer(self.fusion_model.w_emb)

        # answer models
        # Mapping-based methods
        if args.answer_model == 'MLP':
            self.answer_net = MLP(args).cuda()
        elif args.answer_model == 'CLS':
            self.answer_net = SimpleClassifier(
                args.embedding_size, int(args.hidden_size/4), self.dataset.num_answers, 0.5).cuda()

        # # Classifier-based methods
        # self.answer_net = SimpleClassifier(
        #     args.embedding_size, 2 * args.hidden_size, self.dataset.num_answers, 0.5).cuda()

    def _update_best_result(self, max_acc, metrics):
        max_acc[3] = metrics.acc_all
        max_acc[2] = metrics.acc_10
        max_acc[1] = metrics.acc_3
        max_acc[0] = metrics.acc_1

    def _save_model(self, model, function):
        assert function == "fusion" or function == "embedding"
        target = self.args.space_name
        model_name = type(model).__name__
        save_path = os.path.join(self.args.model_save_path, function)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(
            save_path, f'{target}_{model_name}.pkl')

        torch.save(model.state_dict(), save_path)
        return save_path

    def _load_model(self, model, function):
        assert function == "fusion" or function == "embedding"

        # support entity mapping
        target = self.args.space_name
        model_name = type(model).__name__
        save_path = os.path.join(self.args.model_save_path, function)
        save_path = os.path.join(
            save_path, f'{target}_{model_name}.pkl')

        model.load_state_dict(torch.load(save_path))
        print(f"loading {function} model done!")
