import argparse
import tools
from OpenKE.module.model import TransE
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/workspace/datasets/KVQA")
    parser.add_argument("--output_dir", type=str, default="./data/KVQA")
    # parser.add_argument("--kge_ckpt", type=str,
    #                     default="./data/KVQA/transe_1024.ckpt")
    parser.add_argument("--kge_dim", type=int, default=300)

    return parser.parse_args()


def main():
    args = parsing_argument()
    st = time.time()
    fact_set, qa_set, ent_count, rel_count, q_word_count, ans_count = tools.load_facts(
        args.data_root
    )
    print("load facts from raw data:", time.time() - st)

    tools.save_sets(args.output_dir, fact_set, qa_set)

    st = time.time()
    entity2id, relation2id, answer2id, word2id = tools.make_dict_files(
        args.output_dir, ent_count, rel_count, q_word_count, ans_count
    )
    print("make_dict_files:", time.time() - st)

    st = time.time()
    fact_set, qa_set = tools.load_sets(args.output_dir)
    print("load facts from pickle:", time.time() - st)

    st = time.time()
    tools.make_train_files(args.output_dir, fact_set,
                           qa_set, entity2id, relation2id, answer2id, word2id)
    print("make_train_files:", time.time() - st)

    tools.generate_addictive_files(args.output_dir)

    kge_ckpt = args.output_dir + '/transe_{}.ckpt'.format(args.kge_dim)
    transe = tools.train_KGE_transe(
        args.output_dir, kge_ckpt, args.kge_dim)

    # define the model
    transe = TransE(
        ent_tot=len(entity2id),  # len(ent_count), # 337941
        rel_tot=len(relation2id),  # len(rel_count), # 124
        dim=args.kge_dim,
        p_norm=1,
        norm_flag=True,
    )
    tools.test_KGE_transe(transe, args.output_dir, kge_ckpt)


if __name__ == "__main__":
    main()
