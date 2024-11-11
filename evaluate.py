from diversify.alg import alg, modelopera
from diversify.utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from diversify.datautil.getdataloader_single import get_act_dataloader


def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    if args.latent_domain_num < 6:
        args.batch_size = 32*args.latent_domain_num
    else:
        args.batch_size = 16*args.latent_domain_num

    _, _, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()

    valid_acc = modelopera.accuracy(algorithm, valid_loader, None)
    target_acc = modelopera.accuracy(algorithm, target_loader, None)

    print(f'Valid acc: {valid_acc:.4f}, Target acc: {target_acc:.4f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
