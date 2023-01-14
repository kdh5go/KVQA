import os
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from data.dataloader import KVQA_EvalDataset
from utils import Metrics, freeze_layer
from model.answer_mlp import MLP
from model import SimpleClassifier
from model import fusion_net
from utils.tool import find_tail_by_KG


class Evaluator:
    def __init__(self, args):
        # prepare for: data , model, loss fuction, optimizer

        self.args = args
        self.fusion_model = {'relation': Variable(), 'answer': Variable()}
        self.answer_net = {'relation': Variable(), 'answer': Variable()}
        self.KG_weight = self.args.kg_weight

        self.val_metrics = Metrics()

        # data load
        self.dataset = KVQA_EvalDataset(args)
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.TRAIN.batch_size,
            shuffle=False,  # only shuffle the data in training
            pin_memory=True,
            num_workers=args.TRAIN.data_workers,
        )
        self.KG_facts = self.dataset.KG_facts
        self.num_answers = self.dataset.num_answers
        self.kgeid2answerid = self.dataset.kgeid2answerid

        self._model_choice(args, 'relation')
        self._model_choice(args, 'answer')

        self._load_model(self.fusion_model, "fusion", 'relation', 'MLP')
        self._load_model(self.answer_net, "embedding", 'relation', 'CLS')
        self._load_model(self.fusion_model, "fusion", 'answer', 'MLP')
        self._load_model(self.answer_net, "embedding", 'answer', 'CLS')
        # print("Model's state_dict:")

        # for param_tensor in self.fusion_model['answer'].state_dict():
        #     print(param_tensor, "\t", self.fusion_model['answer'].state_dict()[
        #           param_tensor])

        self.KGE_space = torch.from_numpy(self.dataset.KGE_answer).cuda()

    def eval(self):
        tq = tqdm(self.loader, desc='Eval', ncols=0)
        correct = 0
        correct_choice = 0
        count = 0
        cc = 0
        max_dist = 0
        max_dist2 = 0
        for poi_id, poi, question_features, answers, answers_id, q_len, choices in tq:
            poi = Variable(poi.float()).cuda()
            question_features = Variable(question_features).cuda()
            answers = Variable(answers).cuda()
            q_len = Variable(q_len).cuda()
            poi_id = poi_id.detach().numpy()
            answers_id = answers_id.detach().numpy()

            fusion_embedding_rel = self.fusion_model['relation'](
                poi, question_features, q_len)
            predicts_rel = self.answer_net['relation'](fusion_embedding_rel)
            predicts_rel = torch.argmax(predicts_rel, dim=1)
            predicts_rel = predicts_rel.detach().cpu().numpy()

            mask = np.zeros((len(predicts_rel), self.num_answers),
                            dtype=np.float32)
            for i, predict in enumerate(predicts_rel):
                tail = find_tail_by_KG(self.KG_facts, poi_id[i], predict)
                if tail is not None:
                    for t in tail:
                        if t in self.kgeid2answerid.keys():
                            mask[i, self.kgeid2answerid[t]] = self.KG_weight
                        else:
                            a = 1

            fusion_embedding = self.fusion_model['answer'](
                poi, question_features, q_len)

            if self.args.answer_model_ans == 'CLS':
                # TODO: Normalization?
                predicts = self.answer_net['answer'](fusion_embedding)
                # arg_predicts = torch.argmax(
                #     predicts, dim=1).detach().cpu().numpy()
                # arg_answer = torch.argmax(
                #     answers_onehot, dim=1).detach().cpu().numpy()
                # a = 1
            else:
                # Mapping-based methods
                fusion_embedding_ = fusion_embedding.unsqueeze(dim=0)
                KGE_space_ = self.KGE_space.unsqueeze(dim=0)
                predicts = torch.cdist(fusion_embedding_, KGE_space_, p=2)
                predicts = predicts.squeeze()
                predicts = -predicts
            predicts = predicts.detach().cpu().numpy()

            max_dist += np.average(np.min(predicts, axis=1))

            cc += 1

            # final_predicts = -predicts + mask
            final_predicts = predicts + mask
            final_answers = np.argmax(final_predicts, axis=1)
            correct += (final_answers == answers_id).sum()
            count += len(final_answers)

            candidates = np.zeros((len(predicts_rel), 5),
                                  dtype=np.float32)
            for i, pre in enumerate(final_predicts):
                candidates[i, :] = pre[choices[i]]

            final_choice = np.argmax(candidates, axis=1)
            correct_choice += (final_choice == 0).sum()
        # print(max_dist / cc)
        # print(max_dist2 / cc)
        print('Accuracy(All):', correct / count)
        print('Accuracy(5 choices):', correct_choice / count)

    def _model_choice(self, args, space):
        assert args.fusion_model in ['SAN', 'MLP']
        # models api
        self.fusion_model[space] = getattr(fusion_net, args.fusion_model)(
            args, self.dataset.num_tokens).cuda()

        # answer models
        # Mapping-based methods
        answer_model_type = {
            'relation': args.answer_model_rel, 'answer': args.answer_model_ans}

        if space == 'relation':
            num_answers = self.dataset.num_answers_rel
        else:
            num_answers = self.dataset.num_answers

        if answer_model_type[space] == 'MLP':
            self.answer_net[space] = MLP(args).cuda()
        elif answer_model_type[space] == 'CLS':
            self.answer_net[space] = SimpleClassifier(
                args.embedding_size, int(args.hidden_size/4), num_answers, 0.5).cuda()

    def _load_model(self, model, function, space, model_type):
        assert function == "fusion" or function == "embedding"

        if model_type == 'CLS':
            model_name = 'SimpleClassifier'
        else:
            model_name = model_type
        dim = self.args.KG_feature_dim
        # support entity mapping
        save_path = os.path.join(self.args.model_save_path, function)
        save_path = os.path.join(
            save_path, f'{space}_{model_name}_{dim}.pkl')

        model[space].load_state_dict(torch.load(save_path), strict=True)
        print(f"loading {save_path} model done!")
        model[space].eval().cuda()
