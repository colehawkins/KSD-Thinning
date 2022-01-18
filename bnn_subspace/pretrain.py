"""
Run SGD training for FRN Resnet
"""
import argparse
import torch
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR
from ray import tune
import utils

MOMENTUM_DECAY = 0.9


def train_epoch(train_loader, optimizer, net, epoch, smoke_test=False):
    """ Run one training epoch """
    net.train()

    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = -utils.loss.log_likelihood_fn(
            net=net, batch=data, mean_reduce=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        if smoke_test and i > 100:
            return total_loss

    return total_loss


def run_from_dict(params):

    args = argparse.Namespace()

    for key, val in params.items():
        setattr(args, key, val)

    main(args)


def main(args):

    print(args)

    if getattr(args, 'smoke_test', False):
        print("Short run smoke test")
        setattr(args, 'num_epochs', 5)

    torch.manual_seed(args.seed)
    assert torch.cuda.is_available()
    
    #build dataloaders
    train_loader, test_loader, _, _, extras = utils.data.get_train_test_loaders(
        batch_size=args.batch_size,dataset=args.problem_setting)

    net=utils.models.get_model(problem_setting=args.problem_setting,TEXT=extras[0] if args.problem_setting=='imdb' else None)
    swa_net = AveragedModel(net)
    net.cuda()
    swa_net.cuda()

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          weight_decay=args.wd,
                          momentum=MOMENTUM_DECAY)

    swa_start = int(args.num_epochs * 0.75)

    if getattr(args, "resume_from_checkpoint", None):

        net_checkpoint, optimizer_checkpoint, start_epoch = utils.save.load_net_and_opt_checkpoint(
            args.resume_from_checkpoint)

        #doesn't work for loading SWA model
        assert start_epoch < swa_start

        net.load_state_dict(net_checkpoint)
        optimizer.load_state_dict(optimizer_checkpoint)

        utils.optimizers.convert_to_cuda(optimizer)

    else:
        start_epoch = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=swa_start,
                                                     eta_min=args.lr / 10.0,
                                                     last_epoch=start_epoch -
                                                     1)


    for epoch in range(start_epoch, args.num_epochs):

        total_loss = train_epoch(train_loader=train_loader,
                                 optimizer=optimizer,
                                 net=net,
                                 epoch=epoch,
                                 smoke_test=getattr(args, 'smoke_test', False))

        swa_stage = epoch > swa_start
        test_model_name = 'SWA Model' if swa_stage else 'Non-SWA Model'

        if swa_stage:
            swa_net.update_parameters(net)
        else:
            scheduler.step()

        #only evaluate swa net after starting model averaging
        test_acc, _ = utils.loss.evaluate_fn(test_loader,
                                             swa_net if swa_stage else net)

        print("Epoch {}".format(epoch))
        print("\t{} Average loss: {:.2f}".format(
            test_model_name, total_loss / len(train_loader)))
        print("\tTest accuracy: {:.4f}".format(100.0 * test_acc))
        if tune.is_session_enabled():
            tune.report(accuracy=test_acc, iterations=epoch)

        #save on interval and at last epoch
        if args.save_dir is not None and (epoch % args.save_every == 0
                                          or epoch == args.num_epochs - 1):
            checkpoint_name = "epoch_{}_seed_{}".format(epoch,args.seed)
            utils.save.save_checkpoint(model_state_dict=swa_net.state_dict() if swa_stage else net.state_dict(),
                                       optimizer_state_dict=optimizer.state_dict(),
                                       epoch=epoch,
                                       checkpoint_dir=args.save_dir,
                                       seed=args.seed,
                                       checkpoint_name=checkpoint_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem-setting', type=str, choices=['cifar10','imdb'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--save-dir',
                        type=str,
                        default='/data/frn_checkpoints/pretrain')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
