import torch
from torch.autograd import Variable
from configs import cfg
from data.dataloader import KVQA_Dataset

if __name__ == '__main__':
	cfg = cfg()
	dataset = KVQA_Dataset(cfg)
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=4,
		shuffle=True,  # only shuffle the data in training
		pin_memory=True,
		num_workers=4,
	)
	
	for poi, question_features, answers, q_len in loader:
		poi = Variable(poi.float()).cuda()
		
		question_features = Variable(question_features).cuda()
		answers = Variable(answers).cuda()
		q_len = Variable(q_len).cuda()
		a = 1
	