"""Measure whether stage-2 compensation enlarges the identity margin on the SYSU-MM01 test set.

For each model, forward hooks read the stage-2 main feature F^m_2 (output of layerATT1) and the
compensation 0.3 * NAF2(X) (+ icg2 output for FCG models) from the normal eval forward pass:
    pre  = GAP(F^m_2)            post = GAP(F^m_2 + comp_2)
These are exactly the embeddings DCL compares during training (model_idfc.py, return_aux branch).

Per infrared query q (cross-modal, gallery = visible, same camera rule as eval_sysu):
    mu(q) = min_{g in P(q)} cos(q, g) - max_{g in N(q)} cos(q, g)
    delta(q) = mu_post(q) - mu_pre(q)
Reported per model (averaged over gallery trials):
    mean mu_pre / mu_post / delta, fraction of queries with delta > 0 and delta >= m (DCL target),
    and Rank-1 / mAP / mINP of the stage-2 pre / post features.

Usage (one --model per checkpoint, name:idfc_method:path):
    python tools/measure_dcl_margin.py --data_path /path/to/sysu-mm01/ \
        --model baseline:baseline:result/sysu_baseline_p4_n4_lr_0.1_seed_0_latest.t \
        --model dfc:dcl:result/dcl_l005/sysu_dcl_p4_n4_lr_0.1_seed_0_latest.t
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from data.data_loader import TestData
from data.data_manager import process_query_sysu, process_gallery_sysu
from engine.eval_metrics import eval_sysu
from models.model_idfc import embed_net

parser = argparse.ArgumentParser(description='Stage-2 compensation margin analysis (SYSU-MM01)')
parser.add_argument('--data_path', default='./dataset/sysu-mm01/', type=str)
parser.add_argument('--model', action='append', required=True,
                    help='name:idfc_method:checkpoint_path, idfc_method in baseline|icg|dcl|idfc')
parser.add_argument('--mode', default='all', choices=['all', 'indoor'])
parser.add_argument('--trials', default=10, type=int)
parser.add_argument('--dcl_margin', default=0.02, type=float)
parser.add_argument('--test-batch', default=64, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--out', default='result/dcl_margin_analysis.json', type=str)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build(method, path):
    net = embed_net(395, no_local='on', gm_pool='on', arch='resnet50',
                    use_icg=method in ('icg', 'idfc'))
    ckpt = torch.load(path, map_location='cpu')
    net.load_state_dict(ckpt['net'])
    print('loaded {} (epoch {})'.format(path, ckpt.get('epoch')))
    return net.to(device).eval()


def extract(net, loader, modal):
    """Return L2-normalized stage-2 pre / post embeddings (GAP over H x W)."""
    store = {}
    hooks = [net.layerATT1.register_forward_hook(lambda m, i, o: store.__setitem__('main', o)),
             net.NAF2.register_forward_hook(lambda m, i, o: store.__setitem__('naf', o))]
    if net.use_icg:
        hooks.append(net.icg2.register_forward_hook(lambda m, i, o: store.__setitem__('icg', o)))
    pre, post = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            net(x, x, x, x, modal)
            comp = store['naf'] * 0.3 + (store['icg'] if net.use_icg else 0)
            pre.append(F.normalize(store['main'].mean((2, 3)).float(), dim=1).cpu())
            post.append(F.normalize((store['main'] + comp).mean((2, 3)).float(), dim=1).cpu())
    for h in hooks:
        h.remove()
    return torch.cat(pre).numpy(), torch.cat(post).numpy()


def margins(q, g, q_ids, g_ids, q_cams, g_cams):
    sim = q @ g.T
    pos = q_ids[:, None] == g_ids[None, :]
    keep = ~((q_cams[:, None] == 3) & (g_cams[None, :] == 2))  # same rule as eval_sysu
    pos_ok, neg_ok = pos & keep, (~pos) & keep
    valid = pos_ok.any(1) & neg_ok.any(1)
    hp = np.where(pos_ok, sim, 2.0).min(1)
    hn = np.where(neg_ok, sim, -2.0).max(1)
    return (hp - hn)[valid], sim


def loader_for(img, label):
    ds = TestData(img, label, transform=transform_test, img_size=(192, 384))
    return data.DataLoader(ds, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


q_img, q_ids, q_cams = process_query_sysu(args.data_path, mode=args.mode)
q_loader = loader_for(q_img, q_ids)
galleries = []
for t in range(args.trials):
    g_img, g_ids, g_cams = process_gallery_sysu(args.data_path, mode=args.mode, trial=t)
    galleries.append((loader_for(g_img, g_ids), g_ids, g_cams))

results = {}
for spec in args.model:
    name, method, path = spec.split(':', 2)
    start = time.time()
    net = build(method, path)
    q_pre, q_post = extract(net, q_loader, modal=2)
    acc = {k: [] for k in ('mu_pre', 'mu_post', 'delta', 'frac_pos', 'frac_target',
                           'r1_pre', 'r1_post', 'map_pre', 'map_post', 'minp_pre', 'minp_post')}
    for g_loader, g_ids, g_cams in galleries:
        g_pre, g_post = extract(net, g_loader, modal=1)
        mu_pre, s_pre = margins(q_pre, g_pre, q_ids, g_ids, q_cams, g_cams)
        mu_post, s_post = margins(q_post, g_post, q_ids, g_ids, q_cams, g_cams)
        d = mu_post - mu_pre
        acc['mu_pre'].append(mu_pre.mean())
        acc['mu_post'].append(mu_post.mean())
        acc['delta'].append(d.mean())
        acc['frac_pos'].append((d > 0).mean())
        acc['frac_target'].append((d >= args.dcl_margin).mean())
        for tag, s in (('pre', s_pre), ('post', s_post)):
            cmc, mAP, mINP = eval_sysu(-s, q_ids, g_ids, q_cams, g_cams)
            acc['r1_' + tag].append(cmc[0])
            acc['map_' + tag].append(mAP)
            acc['minp_' + tag].append(mINP)
    results[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    results[name].update(method=method, checkpoint=path, mode=args.mode, trials=args.trials)
    print('{} done in {:.1f}s'.format(name, time.time() - start))
    del net
    torch.cuda.empty_cache()

print('\nStage-2 compensation analysis (SYSU-MM01 {}, IR query -> VIS gallery, {} trials)'.format(
    args.mode, args.trials))
print('{:<12}{:>9}{:>9}{:>9}{:>10}{:>11}{:>16}{:>16}'.format(
    'model', 'mu_pre', 'mu_post', 'delta', 'delta>0', 'delta>=m', 'R1 pre->post', 'mAP pre->post'))
for name, r in results.items():
    print('{:<12}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.1f}%{:>10.1f}%{:>8.2f}->{:<6.2f}{:>8.2f}->{:<6.2f}'.format(
        name, r['mu_pre'], r['mu_post'], r['delta'], 100 * r['frac_pos'], 100 * r['frac_target'],
        100 * r['r1_pre'], 100 * r['r1_post'], 100 * r['map_pre'], 100 * r['map_post']))

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(results, f, indent=2)
print('saved to', args.out)
