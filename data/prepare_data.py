import argparse
import tools
from OpenKE.module.model import TransE
import time

def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/workspace/datasets/KVQA')
    parser.add_argument('--output_dir', type=str, default='./data/KVQA')
    parser.add_argument('--kge_ckpt', type=str,
                        default='./data/KVQA/transe.ckpt')
    return parser.parse_args()


def main():
    args = parsing_argument()
    st = time.time()
    fact_set, qa_set, ent_count, rel_count, q_word_count, ans_count = tools.load_facts(args.data_root)
    print('load_facts:', time.time() - st)
    st = time.time()
    tools.make_dict_files(args.output_dir, fact_set, qa_set, ent_count, rel_count, q_word_count, ans_count)
    print('make_dict_files:', time.time() - st)
    # tools.generate_addictive_files(args.output_dir)
    # tools.train_KGE_transe(args.output_dir, args.kge_ckpt)

    # define the model
    transe = TransE(
        ent_tot=337941,  # len(ent_count), # 337941
        rel_tot=124,  # len(rel_count), # 124
        dim=300, p_norm=1, norm_flag=True)

    transe.load_checkpoint(args.kge_ckpt)


if __name__ == '__main__':
    main()
