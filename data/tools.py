import json
import os
from OpenKE.config import Trainer
from OpenKE.module.model import TransE
from OpenKE.module.loss import MarginLoss
from OpenKE.module.strategy import NegativeSampling
from OpenKE.data import TrainDataLoader, TestDataLoader


def extract_QA(annotations):
   for anno in annotations:
       question = anno['question']
       answer = anno['answer']
       

def extract_facts_in_Triple(annotations, ent_count, rel_count):
    facts = [] 
    for anno in annotations:
        triple = anno['triple']
        e1, r, e2 = triple[0], triple[1], triple[2]            
        elem = e2.split(',')
        e2 = ''            
        for i in range(len(elem) - 1):
            e2 += (elem[i].strip() + '/')           
        
        e2 += elem[-1].strip()         
        facts.append([e1, e2, r])
        for e in [e1, e2]:                        
            if not e in ent_count:
                ent_count[e] = 1
            else:
                ent_count[e] += 1
        if not r in rel_count:
            rel_count[r] = 1
        else:
            rel_count[r] += 1
    return facts

def extract_facts_in_QA(annotations, ent_count, rel_count):
    facts = []        
    for anno in annotations:
        triple = anno['fact']['triple']
        triple = triple.split(',')
        e1, r = triple[0], triple[1]
        e2 = ''
        answer_entNum = 1
        for i in range(len(triple)-3):
            e2 += (triple[i + 2].strip() + '/')
            answer_entNum += 1
        e2 += triple[-1].strip()               
        facts.append([e1, e2, r])    
        for e in [e1, e2]:                        
            if not e in ent_count:
                ent_count[e] = 1
            else:
                ent_count[e] += 1
        if not r in rel_count:
            rel_count[r] = 1
        else:
            rel_count[r] += 1
    return facts


def load_facts(data_root):
    fact_set = {'Triple': [], 'QA':[]}
    
    extract_facts = {'QA': extract_facts_in_QA,
                    'Triple': extract_facts_in_Triple}    
    ent_count = dict()
    rel_count = dict()
    # ans_count = dict()
    
    for data_type in ['QA', 'Triple']:    
        regions = os.listdir(os.path.join(data_root, data_type))    
        for region in regions:
            if '.zip' in region:
                continue 
            states = os.listdir(os.path.join(data_root, data_type, region))
            for state in states:
                types = os.listdir(os.path.join(data_root, data_type, region, state))
                for ty in types:
                    files = os.listdir(os.path.join(data_root, data_type, region, state, ty))
                    for fn in files:
                        filepath = os.path.join(data_root, data_type, region, state, ty, fn)
                        with open(filepath, 'r', encoding='utf-8-sig') as f:
                            data = json.load(f)
                            
                        if data_type == 'QA':
                            annotations = data['annotations']['question']
                        else:
                            annotations = data['triples']
                        
                        facts = extract_facts[data_type](annotations, ent_count, rel_count)
                        for fact in facts:
                            fact_set[data_type].append(fact)
                            
                            
    ent_count = {k: v for k, v in sorted(ent_count.items(), key=lambda item: -item[1])}
    rel_count = {k: v for k, v in sorted(rel_count.items(), key=lambda item: -item[1])}
    # ans_count = {k: v for k, v in sorted(ans_count.items(), key=lambda item: item[0])}
    
    return fact_set, ent_count, rel_count


def normalize_count(count):    
    sum = 0
    dist = dict()
    for wn in count:
        sum += count[wn]
    for wn in count:
        dist[wn] = count[wn] / sum        
    return dist


def make_dict_files(output_dir, fact_set, ent_count, rel_count):
    entity2id = dict()
    out_fn = os.path.join(output_dir, 'entity2id.txt')
    with open(out_fn, 'w') as f:
        f.write('{}\n'.format(len(ent_count)))    
        for i, k in enumerate(ent_count):
            f.write('{}\t{}\n'.format(k, i))
            entity2id[k] = i

    relation2id = dict()
    out_fn = os.path.join(output_dir, 'relation2id.txt')
    with open(out_fn, 'w') as f:
        f.write('{}\n'.format(len(rel_count)))    
        for i, k in enumerate(rel_count):
            f.write('{}\t{}\n'.format(k, i))
            relation2id[k] = i

    out_fn = os.path.join(output_dir, 'KGE_train2id.txt')
    with open(out_fn, 'w') as f:
        num = len(fact_set['Triple'])
        print('KGE train fact num :', num)
        f.write('{}\n'.format(num))    
        for fact in fact_set['Triple']:
            f.write('{}\t{}\t{}\n'.format(entity2id[fact[0]], entity2id[fact[1]], relation2id[fact[2]]))

    out_fn = os.path.join(output_dir, 'Fact_train2id.txt')
    with open(out_fn, 'w') as f:
        num = len(fact_set['QA'])
        print('QA train fact num :', num)
        f.write('{}\n'.format(num))    
        for fact in fact_set['QA']:
            f.write('{}\t{}\t{}\n'.format(entity2id[fact[0]], entity2id[fact[1]], relation2id[fact[2]]))  


def generate_addictive_files(output_dir):
	lef = {}
	rig = {}
	rellef = {}
	relrig = {}

	triple = open(os.path.join(output_dir, "KGE_train2id.txt"), "r")
	valid = open(os.path.join(output_dir, "Fact_train2id.txt"), "r")
	test = open(os.path.join(output_dir, "Fact_train2id.txt"), "r")

	tot = (int)(triple.readline())
	for i in range(tot):
		content = triple.readline()
		h,t,r = content.strip().split()
		if not (h,r) in lef:
			lef[(h,r)] = []
		if not (r,t) in rig:
			rig[(r,t)] = []
		lef[(h,r)].append(t)
		rig[(r,t)].append(h)
		if not r in rellef:
			rellef[r] = {}
		if not r in relrig:
			relrig[r] = {}
		rellef[r][h] = 1
		relrig[r][t] = 1

	tot = (int)(valid.readline())
	for i in range(tot):
		content = valid.readline()
		h,t,r = content.strip().split()
		if not (h,r) in lef:
			lef[(h,r)] = []
		if not (r,t) in rig:
			rig[(r,t)] = []
		lef[(h,r)].append(t)
		rig[(r,t)].append(h)
		if not r in rellef:
			rellef[r] = {}
		if not r in relrig:
			relrig[r] = {}
		rellef[r][h] = 1
		relrig[r][t] = 1

	tot = (int)(test.readline())
	for i in range(tot):
		content = test.readline()
		h,t,r = content.strip().split()
		if not (h,r) in lef:
			lef[(h,r)] = []
		if not (r,t) in rig:
			rig[(r,t)] = []
		lef[(h,r)].append(t)
		rig[(r,t)].append(h)
		if not r in rellef:
			rellef[r] = {}
		if not r in relrig:
			relrig[r] = {}
		rellef[r][h] = 1
		relrig[r][t] = 1

	test.close()
	valid.close()
	triple.close()

	f = open(os.path.join(output_dir, "type_constrain.txt"), "w")
	f.write("%d\n"%(len(rellef)))
	for i in rellef:
		f.write("%s\t%d"%(i,len(rellef[i])))
		for j in rellef[i]:
			f.write("\t%s"%(j))
		f.write("\n")
		f.write("%s\t%d"%(i,len(relrig[i])))
		for j in relrig[i]:
			f.write("\t%s"%(j))
		f.write("\n")
	f.close()

	# rellef = {}
	# totlef = {}
	# relrig = {}
	# totrig = {}
	# # lef: (h, r)
	# # rig: (r, t)
	# for i in lef:
	# 	if not i[1] in rellef:
	# 		rellef[i[1]] = 0
	# 		totlef[i[1]] = 0
	# 	rellef[i[1]] += len(lef[i])
	# 	totlef[i[1]] += 1.0

	# for i in rig:
	# 	if not i[0] in relrig:
	# 		relrig[i[0]] = 0
	# 		totrig[i[0]] = 0
	# 	relrig[i[0]] += len(rig[i])
	# 	totrig[i[0]] += 1.0

	# s11=0
	# s1n=0
	# sn1=0
	# snn=0
	# f = open(os.path.join(output_dir, "Fact_train2id.txt"), "r")
	# tot = (int)(f.readline())
	# for i in range(tot):
	# 	content = f.readline()
	# 	h,t,r = content.strip().split()
	# 	rign = rellef[r] / totlef[r]
	# 	lefn = relrig[r] / totrig[r]
	# 	if (rign < 1.5 and lefn < 1.5):
	# 		s11+=1
	# 	if (rign >= 1.5 and lefn < 1.5):
	# 		s1n+=1
	# 	if (rign < 1.5 and lefn >= 1.5):
	# 		sn1+=1
	# 	if (rign >= 1.5 and lefn >= 1.5):
	# 		snn+=1
	# f.close()


	# f = open(os.path.join(output_dir, "Fact_train2id.txt"), "r")
	# f11 = open(os.path.join(output_dir, "1-1.txt"), "w")
	# f1n = open(os.path.join(output_dir, "1-n.txt"), "w")
	# fn1 = open(os.path.join(output_dir, "n-1.txt"), "w")
	# fnn = open(os.path.join(output_dir, "n-n.txt"), "w")
	# fall = open(os.path.join(output_dir, "Fact_train2id_all.txt"), "w")
	# tot = (int)(f.readline())
	# fall.write("%d\n"%(tot))
	# f11.write("%d\n"%(s11))
	# f1n.write("%d\n"%(s1n))
	# fn1.write("%d\n"%(sn1))
	# fnn.write("%d\n"%(snn))
	# for i in range(tot):
	# 	content = f.readline()
	# 	h,t,r = content.strip().split()
	# 	rign = rellef[r] / totlef[r]
	# 	lefn = relrig[r] / totrig[r]
	# 	if (rign < 1.5 and lefn < 1.5):
	# 		f11.write(content)
	# 		fall.write("0"+"\t"+content)
	# 	if (rign >= 1.5 and lefn < 1.5):
	# 		f1n.write(content)
	# 		fall.write("1"+"\t"+content)
	# 	if (rign < 1.5 and lefn >= 1.5):
	# 		fn1.write(content)
	# 		fall.write("2"+"\t"+content)
	# 	if (rign >= 1.5 and lefn >= 1.5):
	# 		fnn.write(content)
	# 		fall.write("3"+"\t"+content)
	# fall.close()
	# f.close()
	# f11.close()
	# f1n.close()
	# fn1.close()
	# fnn.close()

def train_KGE_transe(datapath, ckptpath):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path = datapath + '/', 
        nbatches = 512,
        threads = 8, 
        sampling_mode = "normal", 
        bern_flag = 1, 
        filter_flag = 1, 
        neg_ent = 25,
        neg_rel = 0)

    # define the model
    transe = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 300, 
        p_norm = 1, 
        norm_flag = True)


    # define the loss function
    model = NegativeSampling(
        model = transe, 
        loss = MarginLoss(margin = 5.0),
        batch_size = train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
    trainer.run()
    transe.save_checkpoint(ckptpath)
