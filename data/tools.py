import json
import os
import pickle
from OpenKE.config import Trainer, Tester
from OpenKE.module.model import TransE
from OpenKE.module.loss import MarginLoss
from OpenKE.module.strategy import NegativeSampling
from OpenKE.data import TrainDataLoader, TestDataLoader
from konlpy.tag import Okt


def extract_QA(annotations, word_count, ans_count):
    okt = Okt()
    q_tokens = []
    answers = []
    for anno in annotations:
        question = anno['question']
        # answer = anno['answer']
        triple = anno['fact']['triple']
        triple = triple.split(',')
        e2 = ''
        answer_entNum = 1
        for i in range(len(triple)-3):
            e2 += (triple[i + 2].strip() + '/')
            answer_entNum += 1
        e2 += triple[-1].strip()
        e2 = e2.replace('\n', ' ')
        e2 = 'Error' if e2 == '' else e2
        answer = e2

        token_Q = okt.pos(question, norm=True, stem=True)
        tokens = []
        for t_q in token_Q:
            if not t_q[1] in ['Punctuation', 'KoreanParticle']:
                if not t_q[0] in word_count:
                    word_count[t_q[0]] = 1
                else:
                    word_count[t_q[0]] += 1
                tokens.append(t_q[0])
        q_tokens.append(tokens)
        answer = answer.split(',')
        ans = ''
        for i in range(len(ans)-1):
            ans += (ans[i].strip() + '/')
        ans += answer[-1].strip()
        answer = ans
        if not answer in ans_count:
            ans_count[answer] = 1
        else:
            ans_count[answer] += 1
        # if e2 != answer:
        # 	print(e2, answer)
        answers.append(answer)
    return q_tokens, answers


def extract_facts_in_Triple(annotations, ent_count, rel_count):
    facts = []
    for anno in annotations:
        triple = anno['triple']
        e1, r, e2 = triple[0], triple[1], triple[2]
        e1 = e1.strip()
        elem = e2.split(',')
        e2 = ''
        for i in range(len(elem) - 1):
            e2 += (elem[i].strip() + '/')

        e2 += elem[-1].strip()
        e2 = e2.replace('\n', ' ')
        e2 = 'Error' if e2 == '' else e2
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
        e1 = e1.strip()
        e2 = ''
        answer_entNum = 1
        for i in range(len(triple)-3):
            e2 += (triple[i + 2].strip() + '/')
            answer_entNum += 1
        e2 += triple[-1].strip()
        e2 = e2.replace('\n', ' ')
        e2 = 'Error' if e2 == '' else e2
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
    fact_set = {'Triple': [], 'QA': []}
    qa_set = {'question_tokens': [], 'answer': []}

    extract_facts = {'QA': extract_facts_in_QA,
                     'Triple': extract_facts_in_Triple}
    ent_count = {}
    rel_count = {}
    word_count = {}
    ans_count = {}

    for data_type in ['QA', 'Triple']:
        regions = os.listdir(os.path.join(data_root, data_type))
        for region in regions:
            if '.zip' in region:
                continue
            states = os.listdir(os.path.join(data_root, data_type, region))
            for state in states:
                types = os.listdir(os.path.join(
                    data_root, data_type, region, state))
                for ty in types:
                    files = os.listdir(os.path.join(
                        data_root, data_type, region, state, ty))
                    for fn in files:
                        filepath = os.path.join(
                            data_root, data_type, region, state, ty, fn)
                        with open(filepath, 'r', encoding='utf-8-sig') as f:
                            data = json.load(f)

                        if data_type == 'QA':
                            annotations = data['annotations']['question']
                            q_tokens, answers = extract_QA(
                                annotations, word_count, ans_count)
                            for q, a in zip(q_tokens, answers):
                                qa_set['question_tokens'].append(q)
                                qa_set['answer'].append(a)
                        else:
                            annotations = data['triples']

                        facts = extract_facts[data_type](
                            annotations, ent_count, rel_count)
                        for fact in facts:
                            fact_set[data_type].append(fact)

    ent_count = {k: v for k, v in sorted(
        ent_count.items(), key=lambda item: -item[1])}
    rel_count = {k: v for k, v in sorted(
        rel_count.items(), key=lambda item: -item[1])}
    word_count = {k: v for k, v in sorted(
        word_count.items(), key=lambda item: -item[1])}
    ans_count = {k: v for k, v in sorted(
        ans_count.items(), key=lambda item: -item[1])}

    return fact_set, qa_set, ent_count, rel_count, word_count, ans_count


def normalize_count(count):
    sum = 0
    dist = {}
    for wn in count:
        sum += count[wn]
    for wn in count:
        dist[wn] = count[wn] / sum
    return dist


def save_element2id(save_dir, filename, element_count):
    element2id = {}
    out_fn = os.path.join(save_dir, filename)
    with open(out_fn, 'w') as f:
        f.write('{}\n'.format(len(element_count)))
        for i, k in enumerate(element_count):
            f.write('{}\t{}\t{}\n'.format(k, i, element_count[k]))
            element2id[k] = i
    return element2id


def save_word2id(save_dir, filename, word_count):
    word2id = {}
    out_fn = os.path.join(save_dir, filename)
    with open(out_fn, 'w') as f:
        f.write('{}\n'.format(len(word_count) + 1))
        f.write('{}\t{}\t{}\n'.format(' ', 0, 0))
        word2id[' '] = 0
        for i, k in enumerate(word_count):
            f.write('{}\t{}\t{}\n'.format(k, i + 1, word_count[k]))
            word2id[k] = i+1
    return word2id


def save_answer2id(save_dir, filename, ans_count, entity2id):
    answer2id = {}
    out_fn = os.path.join(save_dir, filename)
    with open(out_fn, 'w') as f:
        f.write('{}\n'.format(len(ans_count)))
        for i, k in enumerate(ans_count):
            f.write('{}\t{}\t{}\t{}\n'.format(
                k, i, ans_count[k], entity2id[k]))
            answer2id[k] = i
    return answer2id


def load_element2id(save_dir, filename):
    element2id = {}
    out_fn = os.path.join(save_dir, filename)
    with open(out_fn, 'r') as f:
        for i, line in enumerate(f):
            if i > 0:
                line = line.strip().split('\t')
                element2id[line[0]] = line[1]
    return element2id


def save_sets(save_dir, fact_set, qa_set):
    out_fn = os.path.join(save_dir, 'fact_set.pkl')
    with open(out_fn, 'wb') as fw:
        pickle.dump(fact_set, fw)
    out_fn = os.path.join(save_dir, 'qa_set.pkl')
    with open(out_fn, 'wb') as fw:
        pickle.dump(qa_set, fw)


def load_sets(save_dir):
    out_fn = os.path.join(save_dir, 'fact_set.pkl')
    with open(out_fn, 'rb') as fr:
        fact_set = pickle.load(fr)
    out_fn = os.path.join(save_dir, 'qa_set.pkl')
    with open(out_fn, 'rb') as fr:
        qa_set = pickle.load(fr)
    return fact_set, qa_set


def make_dict_files(output_dir, ent_count, rel_count, word_count, ans_count):

    entity2id = save_element2id(output_dir, 'entity2id.txt', ent_count)
    relation2id = save_element2id(output_dir, 'relation2id.txt', rel_count)
    answer2id = save_answer2id(
        output_dir, 'answer2id.txt', ans_count, entity2id)
    word2id = save_word2id(output_dir, 'word2id.txt', word_count)
    return entity2id, relation2id, answer2id, word2id


def save_train_file(save_dir, filename, factset, entity2id, relation2id):
    out_fn = os.path.join(save_dir, filename)
    with open(out_fn, 'w') as f:
        f.write('{}\n'.format(len(factset)))
        for fact in factset:
            f.write('{}\t{}\t{}\n'.format(
                    entity2id[fact[0]], entity2id[fact[1]], relation2id[fact[2]]))


def find_max_len_question(q_tokens):
    max_len_question = 0
    for q_t in q_tokens:
        if len(q_t) > max_len_question:
            max_len_question = len(q_t)
    print('max_len_question', max_len_question)


def make_train_files(output_dir, fact_set, qa_set, entity2id, relation2id, answer2id, word2id):

    # entity2id = load_element2id(output_dir, 'entity2id.txt')
    # relation2id = load_element2id(output_dir, 'relation2id.txt')
    # answer2id = load_element2id(output_dir, 'answer2id.txt')
    # word2id = load_element2id(output_dir, 'word2id.txt')

    save_train_file(output_dir, 'KGE_train2id.txt',
                    fact_set['Triple'], entity2id, relation2id)
    print('KGE train fact num :', len(fact_set['Triple']))

    save_train_file(output_dir, 'Fact_train2id.txt',
                    fact_set['QA'], entity2id, relation2id)
    print('QA train fact num :', len(fact_set['QA']))

    q_tokens = qa_set['question_tokens']
    find_max_len_question(q_tokens)

    out_fn = os.path.join(output_dir, 'Question_train2id.txt')
    with open(out_fn, 'w') as f:
        num = len(q_tokens)
        print('Question num :', num)
        f.write('{}\n'.format(num))
        for q_t in q_tokens:
            txt = ''
            for i, q in enumerate(q_t):
                if i < len(q_t) - 1:
                    txt += str(word2id[q]) + ','
                else:
                    txt += str(word2id[q]) + '\n'
            f.write(txt)

    answer = qa_set['answer']
    out_fn = os.path.join(output_dir, 'Answer_train2id.txt')
    with open(out_fn, 'w') as f:
        num = len(answer)
        print('Answer num :', num)
        f.write('{}\n'.format(num))
        for ans in answer:
            if ans in entity2id:
                f.write('{}\n'.format(answer2id[ans]))
            else:
                print('error', ans)
                # for k in answer2id.keys():
                #     if k in ans or ans in k:
                #         print(k)

    # out_fn = os.path.join(output_dir, 'Question_train2id.txt')
    # with open(out_fn, 'w') as f:
    # 	num = len(q_tokens)
    # 	print('Question num :', num)
    # 	f.write('{}\n'.format(num))
    # 	for q_t in q_tokens:
    # 		for
    # 		f.write('{}\t{}\t{}\n'.format(
    # 			entity2id[fact[0]], entity2id[fact[1]], relation2id[fact[2]]))


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
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(valid.readline())
    for i in range(tot):
        content = valid.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(test.readline())
    for i in range(tot):
        content = test.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
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
    f.write("%d\n" % (len(rellef)))
    for i in rellef:
        f.write("%s\t%d" % (i, len(rellef[i])))
        for j in rellef[i]:
            f.write("\t%s" % (j))
        f.write("\n")
        f.write("%s\t%d" % (i, len(relrig[i])))
        for j in relrig[i]:
            f.write("\t%s" % (j))
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


def test_KGE_transe(model, datapath, ckptpath):
    test_dataloader = TestDataLoader(datapath + '/', "link")
    model.load_checkpoint(ckptpath)
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)


def train_KGE_transe(datapath, ckptpath, kge_dim):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=datapath + '/',
        nbatches=512,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=kge_dim,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader,
                      train_times=1000, alpha=1.0, use_gpu=True)
    trainer.run()
    transe.save_checkpoint(ckptpath)
    return transe
