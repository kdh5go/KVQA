import numpy as np
import torch.utils.data as data
from .OpenKE.module.model import TransE


class KVQA_Dataset(data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.facts = []
        self.questions = []
        self.answers = []
        self.load_data()
        self.kge_model = TransE(
                                ent_tot=337941,  # len(ent_count), # 337941
                                rel_tot=124,  # len(rel_count), # 124
                                dim=300, p_norm=1, norm_flag=True)
        self.kge_model.load_checkpoint(args.kge_ckpt)
        self.ent_embeddings = self.kge_model.ent_embeddings.weight
        self.rel_embeddings = self.kge_model.rel_embeddings.weight
        self.question_max_length = self.args.KVQA.question_max_length
        a  = 1
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
        
        
    def __getitem__(self, item):
        e1 = int(self.facts[item][0])
        poi = self.ent_embeddings[e1].detach().numpy()
        question = self.questions[item][0].split(',')
        question_length = len(question)
        question = np.array(question, dtype=int)
        question = np.pad(question, (0, self.question_max_length - question_length))
        target = int(self.answers[item][0])
        
        
        return poi, question, target, question_length
    
    def __len__(self):
        return len(self.questions)