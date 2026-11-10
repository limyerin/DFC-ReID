from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data.data_loader import SYSUData, RegDBData, LLCMData, TestData
from data.data_manager import *
from engine.eval_metrics import eval_sysu, eval_regdb, eval_llcm
from models.model_idfc import embed_net
from models.idfc_modules import discriminative_compensation_loss
from engine.utils import *
from engine.loss import OriTripletLoss, TripletLoss_WRT, CenterTripletLoss,CenterLoss,KLalignloss_DIMA
from engine.loss import QuarCenterTripletLoss as QuarCenterTripletLoss
from tensorboardX import SummaryWriter

from solver import build_optimizer
from solver import build_lrscheduler
from data import build_transforms

from torch.cuda import amp

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',help='dataset name: regdb or sysu or llcm]')
parser.add_argument('--arch', default='resnet50', type=str,help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int,metavar='s', help='save model every 10 epochs')
parser.add_argument('--optim', default='SGD', type=str, help='SGD,ADM')
parser.add_argument('--model_path', default='result/', type=str, help='model save path')
parser.add_argument('--log_path', default='result/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='result/', type=str, help='log save path')
parser.add_argument('--loss_tri', default='QCT', type=str, help='')
parser.add_argument('--lr_scheduler', default='step',type=str, help='step or consine')
parser.add_argument('--backbone', default='AGW',type=str, help='AGW')
parser.add_argument('--lr', default=0.1, type=float,help='learning rate, 0.00035/0.0001 for adam,sgd 0.1')
parser.add_argument('--workers', default=6, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int,metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.7, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,metavar='t', help='random seed')
parser.add_argument('--data_root', default='./dataset', type=str, help='directory containing sysu-mm01/, RegDB/, LLCM/')
parser.add_argument('--gpu', default='0', type=int,help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--method', choices=['baseline', 'icg', 'dcl', 'idfc'], default='baseline')
parser.add_argument('--lambda_dcl', type=float, default=0.05)
parser.add_argument('--dcl_margin', type=float, default=0.02)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--max_steps', type=int, default=0, help='0=all batches; use 2 for smoke test')
# Use the same --amp/--benchmark setting for every method being compared.
parser.add_argument('--amp', action='store_true', help='mixed precision (fp16) training')
parser.add_argument('--benchmark', action='store_true',
                    help='cudnn.benchmark=True (faster, not bitwise reproducible)')
parser.add_argument('--fast', action='store_true', help='shorthand for --amp --benchmark')
parser.add_argument('--init_from', default='', type=str,
                    help='baseline checkpoint to start from (go/no-go fine-tune); '
                         'continues its epoch and LR schedule, ICG params start from zero-init')
parser.add_argument('--new_lr_mult', default=10.0, type=float,
                    help='with --init_from: LR multiplier for the newly added ICG params')

args = parser.parse_args()
use_icg = args.method in ('icg', 'idfc')
use_dcl = args.method in ('dcl', 'idfc')
if args.backbone != 'AGW':
    raise ValueError('IDFC experiment requires --backbone AGW')
if args.optim != 'SGD':
    raise ValueError('Use --optim SGD for controlled IDFC comparisons')

torch.cuda.set_device(args.gpu)
set_seed(args.seed)
if args.fast:
    args.amp = args.benchmark = True
if args.benchmark:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


dataset = args.dataset
if dataset == 'sysu':
    data_path = os.path.join(args.data_root, 'sysu-mm01') + '/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = os.path.join(args.data_root, 'RegDB') + '/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'llcm':
    data_path = os.path.join(args.data_root, 'LLCM') + '/'
    log_path = args.log_path + 'llcm_log/'
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;

checkpoint_path = os.path.join(args.model_path, '')
log_path = os.path.join(log_path, '')

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_{}_p{}_n{}_lr_{}_seed_{}'.format(
    args.method, args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'SGD':
    suffix = suffix + '_' + args.optim

if args.amp:
    suffix = suffix + '_amp'
if args.benchmark:
    suffix = suffix + '_bm'
if args.init_from:
    if args.resume:
        raise ValueError('Use either --init_from or --resume, not both')
    suffix = suffix + '_init_' + os.path.splitext(os.path.basename(args.init_from))[0]

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt', mode='a')
vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
# normalize
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = build_transforms.test_transforms(
    args.img_h, args.img_w, normalize)
transform_color1 = build_transforms.train_transforms_color1(
    args.img_h, args.img_w, normalize)
transform_color2 = build_transforms.train_transforms_color2(
    args.img_h, args.img_w, normalize)
transform_thermal1 = build_transforms.train_transforms_thermal1(
    args.img_h, args.img_w, normalize)
transform_thermal2 = build_transforms.train_transforms_thermal2(
    args.img_h, args.img_w, normalize)
transform_train = transform_color1, transform_color2, transform_thermal1, transform_thermal2

end = time.time()

if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(
        data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(
        data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, args.img_h,
                         args.img_w, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(
        data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(
        data_path, trial=args.trial, modal='thermal')
    
elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, args.img_h,
                        args.img_w, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(
        data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(
        data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test,
                   img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label,
                    transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(
    gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(
    queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)


print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(
    n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(
    n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')


##############################################

if args.backbone == 'AGW':
    embeding_dim = 2048
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch,
                    use_icg=use_icg, return_aux=use_dcl)
else:
    embeding_dim = 2048
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)

net.to(device)
# cudnn.benchmark stays False (set_seed) for seed-reproducible comparisons,
# unless --benchmark / --fast is given

if args.init_from:
    init_ckpt = torch.load(args.init_from, map_location='cpu')
    missing, unexpected = net.load_state_dict(init_ckpt['net'], strict=False)
    bad_missing = [k for k in missing if not k.startswith('icg')]
    if bad_missing or unexpected:
        raise RuntimeError('init_from mismatch: missing={} unexpected={}'.format(bad_missing, unexpected))
    start_epoch = init_ckpt['epoch']
    print('==> init from {} (epoch {}); {} ICG tensors start from zero-init'.format(
        args.init_from, start_epoch, len(missing)))
    del init_ckpt

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
# criterion_id_s = nn.CrossEntropyLoss(label_smoothing=0.5)
if args.loss_tri == 'QCT':
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = QuarCenterTripletLoss(k_size=4, margin=0.7)

else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)


criterion_id.to(device)
criterion_tri.to(device)


# Single SGD optimizer: the original deep optimizer already contains all trainable
# parameters. The original shallow optimizer is built using a consumed iterator.
# With --init_from, the newly added ICG params get their own group (new_lr_mult x base LR).
new_params = [p for n, p in net.named_parameters() if n.startswith('icg')] if args.init_from else []
ignored = set(map(id, net.bottleneck.parameters())) | set(map(id, net.classifier.parameters())) \
    | set(map(id, new_params))
base_params = [p for p in net.parameters() if p.requires_grad and id(p) not in ignored]
param_groups = [
    {'params': base_params, 'lr': 0.1 * args.lr},
    {'params': list(net.bottleneck.parameters()), 'lr': args.lr},
    {'params': list(net.classifier.parameters()), 'lr': args.lr},
]
if new_params:
    param_groups.append({'params': new_params, 'lr': args.new_lr_mult * 0.1 * args.lr})
optimizer = torch.optim.SGD(param_groups, weight_decay=5e-4, momentum=0.9, nesterov=True)
scdule = build_lrscheduler.adjust_learning_rate(optimizer, args.lr_scheduler)
if args.init_from:
    # advance the schedule to the checkpoint's epoch (optimizer momentum starts fresh)
    for _ in range(start_epoch):
        scdule.step()
    print('==> LR at epoch {}: {}'.format(start_epoch, [g['lr'] for g in optimizer.param_groups]))
scaler = amp.GradScaler(enabled=args.amp)
# Resume optimizer, scheduler, scaler only after they exist.
if len(args.resume) > 0 and os.path.isfile(model_path):
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint:
        scdule.load_state_dict(checkpoint['scheduler'])
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    best_acc = checkpoint.get('best_acc', best_acc)
    best_epoch = checkpoint.get('best_epoch', -1)
else:
    best_epoch = -1


def train(epoch):

    current_lr = scdule.get_lr()[0]
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    dcl_meter = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()

    correct_id = 0

    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input20, input21, label1, label2) in enumerate(trainloader):

        with amp.autocast(enabled=args.amp):

            labels = torch.cat((label1, label1, label2, label2), 0)

            input10 = Variable(input10.cuda())
            input11 = Variable(input11.cuda())
            input20 = Variable(input20.cuda())
            input21 = Variable(input21.cuda())


            labels = Variable(labels.cuda())
            data_time.update(time.time() - end)

            outputs = net(input10, input11, input20, input21)
            if use_dcl:
                feat, out0, feat_pre, feat_post = outputs
            else:
                feat, out0 = outputs

            loss_id = criterion_id(out0, labels)
            loss_tri = criterion_tri(feat, labels)
            loss_dcl = loss_id.new_zeros(())
            if use_dcl:
                loss_dcl = discriminative_compensation_loss(
                    feat_pre, feat_post, labels, margin=args.dcl_margin)
            loss = loss_id + loss_tri + args.lambda_dcl * loss_dcl
            if use_dcl and batch_idx == 0:
                print(
                    f"[DCL CHECK] "
                    f"loss_dcl={loss_dcl.item():.6f}, "
                    f"weighted={args.lambda_dcl * loss_dcl.item():.6f}"
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch}, batch={batch_idx}')

            # Count correct predictions once per batch.
            _, predicted = out0.max(1)
            correct_id += predicted.eq(labels).sum().item()



        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscale once before checking gradients (and before scaler.step).
        scaler.unscale_(optimizer)
        if use_icg and batch_idx == 0:
            found_icg = False
            for name, param in net.named_parameters():
                if 'icg' in name.lower():
                    found_icg = True
                    if param.grad is None:
                        print(f"[ICG GRAD] {name}: None")
                    else:
                        print(
                            f"[ICG GRAD] {name}: "
                            f"{param.grad.norm().item():.8f}"
                        )
            if not found_icg:
                print('[ICG WARNING] No ICG parameters found!')
        scaler.step(optimizer)
        scaler.update()

        # update P
        train_loss.update(loss.item(), labels.size(0))
        id_loss.update(loss_id.item(), labels.size(0))
        tri_loss.update(loss_tri.item(), labels.size(0))
        dcl_meter.update(loss_dcl.item(), labels.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.6f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'IDLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'QCTLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f} '
                  'PeakMem: {:.1f}GB '.format(
                      epoch, batch_idx, len(trainloader), current_lr,
                      100. * correct_id / total,
                      torch.cuda.max_memory_reserved() / 1024 ** 3, batch_time=batch_time,
                      train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))
        if args.max_steps > 0 and batch_idx + 1 >= args.max_steps:
            print(f'Smoke test: stopped after {args.max_steps} steps')
            break
    scdule.step()

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('dcl_loss', dcl_meter.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)



def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    nout = embeding_dim
    gall_feat = np.zeros((ngall, nout))
    gall_feat_att = np.zeros((ngall, nout))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)

            # input = torch.cat((input, input,), 0)
            input = Variable(input.cuda())
            featA, feat_attA = net(input, input, input, input, test_mode[0])

            gall_feat[ptr:ptr + batch_num, :] = featA.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num,
                          :] = feat_attA.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    nout = embeding_dim
    query_feat = np.zeros((nquery, nout))
    query_feat_att = np.zeros((nquery, nout))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            featA, feat_attA = net(input, input, input, input, test_mode[1])

            query_feat[ptr:ptr + batch_num, :] = featA.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num,
                           :] = feat_attA.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(
            -distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label,
                                   gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(
            -distmat_att, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(-distmat, query_label,
                                   gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(
            -distmat_att, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, args.epochs):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label,
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    #print(trainset.cIndex)
    #print(trainset.tIndex)
    print(f"Color indices: {len(trainset.cIndex)}")
    print(f"Thermal indices: {len(trainset.tIndex)}")

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch,
                                  sampler=sampler, num_workers=args.workers, drop_last=True,)

    # training
    train(epoch)

    if args.max_steps == 0:
        latest_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scdule.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'method': args.method,
        }
        torch.save(latest_state, os.path.join(checkpoint_path, suffix + '_latest.t'))

    if args.max_steps > 0:
        print('Smoke test finished. Evaluation and checkpoints skipped.')
        break

    # periodic snapshot, independent of the evaluation schedule
    if (epoch + 1) % args.save_epoch == 0:
        state = {
            'net': net.state_dict(),
            'epoch': epoch + 1,
            'method': args.method,
        }
        torch.save(state, checkpoint_path + suffix +
                   '_epoch_{}.t'.format(epoch + 1))

    if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scdule.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'method': args.method,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))
