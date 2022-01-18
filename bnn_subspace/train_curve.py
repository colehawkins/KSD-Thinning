"""
Run SGD training for FRN Resnet
"""
import argparse
import torch
from torch import optim
from ray import tune
import utils

MOMENTUM_DECAY = 0.9
BATCH_SIZE = 128


def create_curve_module_grads(net):
    #Need to create curve module grads to set later

    loss = sum([x.norm() for _, x in net.midpoint_dict.items()])

    loss.backward()


@torch.no_grad()
def set_curve_module_grads(net, t):
    #Need to set curve module gradients manually
    #Only the base network parameters get gradients
    if t < 0.5:
        coeff = t / 0.5
    else:
        coeff = 1 - (t - 0.5) / 0.5

    for name, param in net.net.named_parameters():

        assert hasattr(param, 'grad') and param.grad is not None

        setattr(net.midpoint_dict[name], 'grad', coeff * param.grad)


def train_epoch(train_loader, optimizer, net, epoch, smoke_test=False):
    """ Run one training epoch """
    #only works on curve model
    assert isinstance(net, utils.models.CurveModel)
    net.train()

    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(train_loader):

        optimizer.zero_grad()
        net.net.zero_grad()

        #evaluate random point on the curve at every batch element
        t = torch.rand(1).to(net.device)
        loss = -utils.loss.log_likelihood_fn(
            net=lambda x: net(x, t), batch=data, mean_reduce=True)

        loss.backward()

        #need to set these based on the backprop through
        set_curve_module_grads(net, t)
        optimizer.step()

        #net.midpoint_dict = net.net.state_dict()
        running_loss += loss.item()
        total_loss += loss.item()
        if smoke_test and i > 10:
            return total_loss

    return total_loss


def main(args):

    print(args)

    if getattr(args, 'smoke_test', False):
        print("Short run smoke test")
        setattr(args, 'num_epochs', 5)

    torch.manual_seed(args.seed)
    assert torch.cuda.is_available()

    train_loader, test_loader, _, _, extras = utils.data.get_train_test_loaders(
        batch_size=args.batch_size,dataset=args.problem_setting)

    base_net=utils.models.get_model(problem_setting=args.problem_setting,TEXT=extras[0] if args.problem_setting=='imdb' else None)

    endpoint_1_dict = utils.save.load_net_and_opt_checkpoint(
        args.endpoint_1)['model_state_dict']

    endpoint_2_dict = utils.save.load_net_and_opt_checkpoint(
        args.endpoint_2)['model_state_dict']

    net = utils.models.construct_curve_model(base_net, endpoint_1_dict,
                                             endpoint_2_dict)

    net.to('cuda')

    #Only optimize the midpoint net parameters
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          weight_decay=args.wd,
                          momentum=MOMENTUM_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.num_epochs,
                                                     eta_min=args.lr / 10.0)
    create_curve_module_grads(net)


    for epoch in range(args.num_epochs):

        total_loss = train_epoch(train_loader=train_loader,
                                 optimizer=optimizer,
                                 net=net,
                                 epoch=epoch,
                                 smoke_test=getattr(args, 'smoke_test', False))

        scheduler.step()

        #only evaluate midpoint with t=0.5
        net.eval()
        test_acc, _ = utils.loss.evaluate_fn(test_loader, net)

        print("Epoch {}".format(epoch))
        print("\tTraining average loss: {:.2f}".format(total_loss /
                                                       len(train_loader)))
        print("\tTest accuracy: {:.4f}".format(100.0 * test_acc))
        if tune.is_session_enabled():
            tune.report(accuracy=test_acc, iterations=epoch)

        #save on interval or on final
        if args.save_dir is not None and (epoch % args.save_every == 0
                                          or epoch == (args.num_epochs - 1)):

            checkpoint_name = "epoch_{}_seed_{}".format(epoch,args.seed)
           
            utils.save.save_checkpoint(
                                checkpoint_dir=args.save_dir,
                                checkpoint_name=checkpoint_name,
                                endpoint_1=net.dict_1,
                                endpoint_2=net.dict_2,
                                midpoint=net.midpoint_dict,
                                epoch=epoch,
                                seed=args.seed
                                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--problem-setting', type=str, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--save-dir',
                        type=str,
                        default='/data/frn_checkpoints/curve')
    parser.add_argument('--endpoint-1',
                        type=str,
                        help='Path to first curve endpoint checkpoint')
    parser.add_argument('--endpoint-2',
                        type=str,
                        help='Path to second curve endpoint checkpoint')
    parser.add_argument('--smoke-test', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
