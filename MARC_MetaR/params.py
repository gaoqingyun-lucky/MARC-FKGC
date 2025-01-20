import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="NELL-One", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-path", "--data_path", default="./NELL", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=1, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=30, type=int)

    args.add_argument("-epo", "--epoch", default=10000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=1000, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)

    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    # --------------------------------------------------------------------------------------------
    args.add_argument('--dims', type=str, default='[1000]')
    args.add_argument('--d_emb_size', type=int, default=10)
    args.add_argument('--norm', type=bool, default=True)
    args.add_argument('--steps', type=int, default=5)
    args.add_argument('--noise_scale', type=float, default=0.1)
    args.add_argument('--noise_min', type=float, default=0.0001)
    args.add_argument('--noise_max', type=float, default=0.02)
    args.add_argument('--sampling_steps', type=int, default=0)
    args.add_argument("-max_neighbor", "--max_neighbor", default=20, type=int)
    args.add_argument("-hop", "--hop", default=2, type=int)
    args.add_argument("--K", default=10, type=int)
    args.add_argument("--g_batch", default=512, type=int)
    # --------------------------------------------------------------------------------------------

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'NELL-One':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50

    params['device'] = torch.device('cuda:'+str(args.device))

    return params


data_dir = {
    'train_tasks_in_train': '/train_tasks_in_train.json',
    'train_tasks': '/train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/dev_tasks.json',

    'rel2candidates_in_train': '/rel2candidates_in_train.json',
    'rel2candidates': '/rel2candidates.json',

    'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
    'e1rel_e2': '/e1rel_e2.json',

    'ent2ids': '/ent2ids',
    'ent2vec': '/entity2vec.TransE',
    'rel2vec': '/relation2vec.TransE',
}
