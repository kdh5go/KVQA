import numpy as np
import pickle
import torch.utils.data as data
from .OpenKE.module.model import TransE


class KVQA_Dataset(data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.facts = []
        self.questions = []
        self.answers = []
        self.answer2id = []
        self.answerid2kgeid = {}
        self.num_tokens = 234
        self.num_answers = 0
        self.imageInfo = {}

        self.load_data()
        self.kge_model = TransE(
            ent_tot=self.args.KVQA.num_entity,  # len(ent_count), # 337941
            rel_tot=self.args.KVQA.num_relation,  # len(rel_count), # 124
            dim=self.args.KG_feature_dim, p_norm=1, norm_flag=True)
        self.kge_model.load_checkpoint(args.kge_ckpt)
        self.ent_embeddings = self.kge_model.ent_embeddings.weight
        self.rel_embeddings = self.kge_model.rel_embeddings.weight
        self.question_max_length = self.args.KVQA.question_max_length
        self.KGE_answer = []
        self.set_KGE_answer()

        if self.args.space_name == 'relation':
            self.num_answers = len(self.rel_embeddings)

    def set_KGE_answer(self):
        for dt in self.answer2id:
            answer_id = int(dt[1])
            entity_id = int(dt[3])
            embedding = self.ent_embeddings[entity_id].detach().numpy()
            self.answerid2kgeid[answer_id] = entity_id
            self.KGE_answer.append(embedding)
        self.KGE_answer = np.array(
            self.KGE_answer, dtype=np.float32)

    def load_data(self):
        with open(self.args.KVQA.fact_path, "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.strip().split('\t')
                    self.facts.append(line)
        with open(self.args.KVQA.question_path, "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.strip().split('\t')
                    self.questions.append(line)
        with open(self.args.KVQA.answer_path, "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.strip().split('\t')
                    self.answers.append(line)
        with open(self.args.KVQA.word2id_path, "r") as f:
            self.num_tokens = int(f.readline())

        with open(self.args.KVQA.answer2id_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    self.num_answers = int(line)
                else:
                    line = line.strip().split('\t')
                    self.answer2id.append(line)
        with open(self.args.KVQA.imageInfo_path, 'rb') as fr:
            self.imageInfo = pickle.load(fr)

    def __getitem__(self, item):

        image_filename = self.imageInfo['image_filename'][item]
        captions = self.imageInfo['captions'][item]

        poi_id = int(self.facts[item][0])
        poi = self.ent_embeddings[poi_id].detach().numpy()
        question = self.questions[item][0].split(',')
        question_length = len(question)
        question = np.array(question, dtype=int)
        question = np.pad(
            question, (0, self.question_max_length - question_length))

        if self.args.space_name == 'answer':
            answer_id = int(self.answers[item][0])
            answer_kgeid = self.answerid2kgeid[answer_id]
            target = self.ent_embeddings[answer_kgeid].detach().numpy()
            target_onehot = np.zeros(self.num_answers)
            target_onehot[answer_id] = 1
        elif self.args.space_name == 'relation':
            rel_id = int(self.facts[item][2])
            target = self.rel_embeddings[rel_id].detach().numpy()
            target_onehot = np.zeros(124)
            target_onehot[rel_id] = 1
        return poi, question, target, target_onehot, question_length

    def __len__(self):
        return len(self.questions)


class KVQA_EvalDataset(data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.facts = []
        self.questions = []
        self.answers = []
        self.answer2id = []
        self.answerid2kgeid = {}
        self.kgeid2answerid = {}
        self.num_tokens = 234
        self.num_answers = 0
        self.imageInfo = {}

        self.load_data()
        self.kge_model = TransE(
            ent_tot=self.args.KVQA.num_entity,  # len(ent_count), # 337941
            rel_tot=self.args.KVQA.num_relation,  # len(rel_count), # 124
            dim=self.args.KG_feature_dim, p_norm=1, norm_flag=True)
        self.kge_model.load_checkpoint(args.kge_ckpt)
        self.ent_embeddings = self.kge_model.ent_embeddings.weight
        self.rel_embeddings = self.kge_model.rel_embeddings.weight
        self.question_max_length = self.args.KVQA.question_max_length
        self.KGE_answer = []
        self.set_KGE_answer()
        self.num_answers_rel = len(self.rel_embeddings)
        # if self.args.space_name == 'relation':
        #     self.num_answers = len(self.rel_embeddings)

        self.KG_facts = {}
        self.num_choices = 5
        self.choices = []
        self.load_KG_facts()
        self.generate_choices()

    def generate_choices(self):
        num_answers = len(self.answer2id)
        for ans in self.answers:
            choices = [int(ans[0])]
            for _ in range(self.num_choices - 1):
                while 1:
                    choice = np.random.randint(0, num_answers)
                    if not choice in choices:
                        choices.append(choice)
                        break
            # choices = np.random.shuffle(choices)
            self.choices.append(choices)

    def load_KG_facts(self):
        with open(self.args.KVQA.KG2id_path, "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.strip().split('\t')
                    line = [int(x) for x in line]

                    if not line[0] in self.KG_facts.keys():
                        self.KG_facts[line[0]] = {}
                        self.KG_facts[line[0]][line[2]] = [line[1]]
                    else:
                        if not line[2] in self.KG_facts[line[0]]:
                            self.KG_facts[line[0]][line[2]] = [line[1]]
                        else:
                            self.KG_facts[line[0]][line[2]].append(line[1])

    def set_KGE_answer(self):
        for dt in self.answer2id:
            answer_id = int(dt[1])
            entity_id = int(dt[3])
            embedding = self.ent_embeddings[entity_id].detach().numpy()
            self.answerid2kgeid[answer_id] = entity_id
            self.kgeid2answerid[entity_id] = answer_id
            self.KGE_answer.append(embedding)
        self.KGE_answer = np.array(
            self.KGE_answer, dtype=np.float32)

    def load_data(self):
        with open(self.args.KVQA.fact_path, "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.strip().split('\t')
                    self.facts.append(line)
        with open(self.args.KVQA.question_path, "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.strip().split('\t')
                    self.questions.append(line)
        with open(self.args.KVQA.answer_path, "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.strip().split('\t')
                    self.answers.append(line)
        with open(self.args.KVQA.word2id_path, "r") as f:
            self.num_tokens = int(f.readline())

        with open(self.args.KVQA.answer2id_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    self.num_answers = int(line)
                else:
                    line = line.strip().split('\t')
                    self.answer2id.append(line)
        with open(self.args.KVQA.imageInfo_path, 'rb') as fr:
            self.imageInfo = pickle.load(fr)

    def __getitem__(self, item):

        image_filename = self.imageInfo['image_filename'][item]
        captions = self.imageInfo['captions'][item]

        poi_id = int(self.facts[item][0])
        poi = self.ent_embeddings[poi_id].detach().numpy()
        question = self.questions[item][0].split(',')
        question_length = len(question)
        question = np.array(question, dtype=int)
        question = np.pad(
            question, (0, self.question_max_length - question_length))

        answer_id = int(self.answers[item][0])
        answer_kgeid = self.answerid2kgeid[answer_id]
        answer = self.ent_embeddings[answer_kgeid].detach().numpy()

        choices = np.array(self.choices[item], dtype=int)

        return poi_id, poi, question, answer, answer_id, question_length, choices

    def __len__(self):
        return len(self.questions)
