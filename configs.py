import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse
import pdb


class cfg():
    def __init__(self):

        # log settings
        self.exp_name = 'mlp'
        self.exp_id = ''
        self.log_dir = './logs'
        self.no_tensorboard = False
        self.random_seed = 1234

        # common settings
        self.now_test = False
        self.space_name = 'answer'
        self.model_save_path = '../KVQA_train'
        self.save_model = True

        # path settings
        self.kg_data_root = '/workspace/datasets/KVQA'
        self.qa_data_root = '/workspace/datasets/QA'
        self.data_output_dir = './data/KVQA'
        self.kge_ckpt = './data/KVQA/transe_300.ckpt'

        # embedding settings
        self.KG_feature_dim = 300
        self.visual_feature_dim = 300  # POI --> KGE
        # self.embedding_size = 512  # embedding dimensionality
        self.embedding_size = 300  # embedding dimensionality
        self.hidden_size = 2 * self.embedding_size  # hidden embedding

        # model settings
        self.fusion_model = 'MLP'
        self.freeze_w2v = False
        self.answer_model = 'CLS'
        self.answer_model_rel = 'CLS'
        self.answer_model_ans = 'CLS'
        self.ans_net_lay = 0

        self.KVQA = edict()
        self.KVQA.fact_path = './data/KVQA/Fact_train2id.txt'
        self.KVQA.question_path = './data/KVQA/Question_train2id.txt'
        self.KVQA.answer_path = './data/KVQA/Answer_train2id.txt'
        self.KVQA.word2id_path = './data/KVQA/word2id.txt'
        self.KVQA.answer2id_path = './data/KVQA/answer2id.txt'
        self.KVQA.KG2id_path = './data/KVQA/KGE_train2id.txt'
        self.KVQA.imageInfo_path = './data/KVQA/imageInfo_train.pkl'

        self.KVQA.question_max_length = 13
        self.KVQA.num_entity = 337612
        self.KVQA.num_relation = 124

        # train params
        self.TRAIN = edict()
        self.TRAIN.epochs = 200
        self.TRAIN.batch_size = 512  # 512
        self.TRAIN.lr = 1e-3  # default Adam lr 1e-3
        self.TRAIN.CosineAnnealing_Tmax = 50
        self.TRAIN.CosineAnnealing_Tmult = 2
        self.TRAIN.lr_decay_step = 30
        self.TRAIN.lr_decay_rate = .90
        self.TRAIN.data_workers = 8  # 10
        self.loss_temperature = 0.001

        self.patience = 30

        # test params
        self.kg_weight = 0

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu_id', default=0, type=int)
        parser.add_argument('--answer_model', default='CLS',
                            choices=['MLP', 'CLS'])
        parser.add_argument('--space_name', default='answer',
                            choices=['relation', 'answer'])
        # parser.add_argument('--finetune', action='store_true')
        # parser.add_argument('--batch_size', default=128, type=int)
        # parser.add_argument('--max_ans', default=500, type=int)  # 3000 300##
        # parser.add_argument('--loss_temperature', default=0.01, type=float)
        # parser.add_argument('--answer_embedding', default='MLP')
        # parser.add_argument('--embedding_size', default=1024, choices=[1024, 300, 512], type=int)
        # parser.add_argument('--epoch', default=800, type=int)
        # # choice model
        # parser.add_argument('--fusion_model', default='SAN', choices=['MLP', 'SAN'])
        # parser.add_argument('--requires_grad', default=0, type=int, choices=[0, 1])
        # # choice class
        # parser.add_argument('--method_choice', default='W2V',
        #                     choices=['CLS', 'W2V', 'KG', 'GAE', 'KG_W2V', 'KG_GAE', 'GAE_W2V', 'KG_GAE_W2V'])
        # parser.add_argument('--ans_fusion', default='Simple_concate',
        #                     choices=['RNN_concate', 'GATE_attention', 'GATE', 'RNN_GATE_attention', 'Simple_concate'])
        # # KG situation
        # parser.add_argument('--KGE', default='TransE',
        #                     choices=['TransE', 'ComplEx', "TransR", "DistMult"])
        # parser.add_argument('--entity_num', default="all", choices=['all', '4302'])

        # parser.add_argument('--data_choice', default='0', choices=['0', '1', '2', '3', '4'])
        # parser.add_argument('--name', default=None, type=str)  # 定义名字后缀

        # parser.add_argument("--no-tensorboard", default=False, action="store_true")
        # parser.add_argument("--exp_name", default="", type=str, required=True, help="Experiment name")
        # parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        # parser.add_argument("--exp_id", default="", type=str, help="Experiment ID")
        # parser.add_argument("--random_seed", default=4567, type=int)
        # parser.add_argument("--freeze_w2v", default=1, type=int, choices=[0, 1])

        # parser.add_argument("--matching_space", default='answer', type=str, choices=['answer', 'poi', 'relation'])

        # parser.add_argument("--now_test", default=0, type=int, choices=[0, 1])
        # parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])

        # parser.add_argument("--joint_test_way", default=0, type=int, choices=[0, 1])
        # parser.add_argument("--top_rel", default=10, type=int)
        # parser.add_argument("--top_fact", default=100, type=int)
        # parser.add_argument("--soft_score", default=10, type=int)  # 10 or 10000
        # parser.add_argument("--mrr", default=0, type=int)
        args = parser.parse_args()
        return args

    def update_train_configs(self, args):
        self.gpu_id = args.gpu_id
        self.answer_model = args.answer_model
        self.space_name = args.space_name
        if self.exp_id == '':
            self.exp_id = self.space_name
    #     self.finetune = args.finetune
    #     self.answer_embedding = args.answer_embedding
    #     self.name = args.name
    #     self.no_tensorboard = args.no_tensorboard
    #     self.exp_name = args.exp_name
    #     self.dump_path = args.dump_path
    #     self.exp_id = args.exp_id
    #     self.random_seed = args.random_seed
    #     self.freeze_w2v = args.freeze_w2v
    #     self.loss_temperature = args.loss_temperature
    #     self.ZSL = args.ZSL

    #     self.matching_space = args.matching_space
    #     self.now_test = args.now_test
    #     self.save_model = args.save_model
    #     self.joint_test_way = args.joint_test_way
    #     self.top_rel = args.top_rel
    #     self.top_fact = args.top_fact
    #     self.soft_score = args.soft_score
    #     self.mrr = args.mrr

    #     if args.ZSL == 1:
    #         print("ZSL setting...")
    #         self.FVQA.test_data_path = self.FVQA.unseen_test_data_path
    #         self.FVQA.train_data_path = self.FVQA.seen_train_data_path

    #     if args.fusion_model == 'UD' or args.fusion_model == 'BAN':
    #         self.FVQA.feature_path = osp.join(self.FVQA.common_data_path, 'fvqa_36.hdf5')
    #         self.FVQA.img_id2idx = osp.join(self.FVQA.common_data_path, 'fvqa36_imgid2idx.pkl')
    #     self.requires_grad = True if args.requires_grad == 1 else False
    #     self.fusion_model = args.fusion_model
    #     self.TRAIN.batch_size = args.batch_size
    #     # self.TRAIN.answer_batch_size = args.answer_batch_size
    #     self.method_choice = args.method_choice
    #     self.ans_fusion = args.ans_fusion
    #     self.embedding_size = args.embedding_size
    #     self.FVQA.data_choice = args.data_choice
    #     self.FVQA.max_ans = args.max_ans
    #     self.TRAIN.epochs = args.epoch
    #     self.FVQA.KGE = args.KGE
    #     self.FVQA.KGE_init = args.KGE_init
    #     self.FVQA.gae_init = args.GAE_init
    #     self.FVQA.entity_num = args.entity_num

    #     if self.fact_map:
    #         self.FVQA.max_ans = 2791
    #     if self.relation_map:
    #         self.FVQA.max_ans = 103

    #     self.TEST.max_answer_index = self.FVQA.max_ans
    #     self.TRAIN.answer_batch_size = self.FVQA.max_ans  # batch size for answer network
    #     self.TRAIN.max_negative_answer = self.FVQA.max_ans

    #     self.FVQA.answer_vocab_path = osp.join(
    #         self.FVQA.common_data_path, 'answer.vocab.fvqa.' + str(self.FVQA.max_ans) + '.json')

    #     if "KG" in self.method_choice:
    #         self.FVQA.relation2id_path = osp.join(self.FVQA.kg_path, "relations_" + self.FVQA.entity_num + ".tsv")
    #         self.FVQA.entity2id_path = osp.join(self.FVQA.kg_path, "entities_" + self.FVQA.entity_num + ".tsv")
    #         if self.KGE_init != "w2v":
    #             self.FVQA.entity_path = osp.join(self.FVQA.kg_path, "fvqa_" +
    #                                              self.FVQA.entity_num + "_" + self.KGE + "_entity.npy")
    #             self.FVQA.relation_path = osp.join(self.FVQA.kg_path, "fvqa_" +
    #                                                self.FVQA.entity_num + "_" + self.KGE + "_relation.npy")
    #         else:
    #             self.FVQA.entity_path = osp.join(self.FVQA.kg_path, "fvqa_" +
    #                                              self.FVQA.entity_num + "_w2v_" + self.KGE + "_entity.npy")
    #             self.FVQA.relation_path = osp.join(self.FVQA.kg_path, "fvqa_" +
    #                                                self.FVQA.entity_num + "_w2v_" + self.KGE + "_relation.npy")
