"""Retrieval visualization: Baseline (left) vs DFC-ReID (right) on SYSU-MM01 (IR query -> VIS gallery).

Each row shows one infrared query in the middle, the top-K visible gallery images of the baseline on the
left (rank 1 next to the query), and the top-K of DFC-ReID on the right (rank 1 next to the query).
Correct matches have a green border, wrong matches a red border.

Ranking follows test_dfc.py exactly: FC (batch-normalized, L2-normalized) features, cosine similarity,
gallery of the given trial, and the eval_sysu camera rule (cam3 queries ignore cam2 gallery images).

Query selection is rule-based and reproducible (state the rule in the figure caption):
  improved : baseline Rank-1 wrong, DFC-ReID Rank-1 correct (random sample with --seed)
  failure  : both Rank-1 wrong (shown below a dashed separator, as a failure case)
  degraded : baseline Rank-1 correct, DFC-ReID Rank-1 wrong
  random   : any query (random sample with --seed)
The counts of each category over all queries are printed and saved, so the caption can report them.

Usage (run on the machine that has the checkpoints and the dataset):
    python plot_retrieval.py --data_path /path/to/sysu-mm01/ \
        --baseline result/sysu_baseline_p4_n4_lr_0.1_seed_0_latest.t \
        --ours result/dcl_l005/sysu_dcl_p4_n4_lr_0.1_seed_0_latest.t \
        --rows improved:4 --rows failure:1 --topk 10
Outputs: figures/retrieval_vis.{pdf,png} and figures/retrieval_vis_selection.json
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 8 point Times New Roman for figure labels (Liberation Serif is the metric-compatible fallback)
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
                     'font.size': 8, 'pdf.fonttype': 42})

from data.data_loader import TestData
from data.data_manager import process_query_sysu, process_gallery_sysu
from models.model_dfc import embed_net

parser = argparse.ArgumentParser(description='Baseline vs DFC-ReID retrieval visualization')
parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--baseline', required=True, type=str, help='baseline checkpoint (.t)')
parser.add_argument('--ours', required=True, type=str, help='DFC-ReID checkpoint (.t)')
parser.add_argument('--baseline_method', default='baseline', choices=['baseline', 'icg', 'dcl', 'idfc'])
parser.add_argument('--ours_method', default='dcl', choices=['baseline', 'icg', 'dcl', 'idfc'])
parser.add_argument('--mode', default='all', choices=['all', 'indoor'])
parser.add_argument('--trial', default=0, type=int, help='gallery split (0-9), same sampling as test_dfc.py')
parser.add_argument('--rows', action='append', default=None,
                    help='category:count, category in improved|better|failure|degraded|random (repeatable)')
parser.add_argument('--queries', default='', type=str,
                    help='comma-separated query indices to show instead of --rows (category "manual")')
parser.add_argument('--topk', default=10, type=int)
parser.add_argument('--num_figs', default=1, type=int,
                    help='number of multi-row figures to save as <out>_<n>.png/.pdf; every row in every figure '
                         'uses a different identity (1: single figure saved as <out>.png/.pdf)')
parser.add_argument('--curate', action='store_true',
                    help='rank candidates by (DFC-ReID hits@topk - baseline hits@topk) and pick the best, '
                         'instead of sampling uniformly at random')
parser.add_argument('--layout', default='side', choices=['side', 'stack'],
                    help='side: Baseline | Query | Ours in one row; stack: query on the left (vertically centered), '
                         'dashed separator, baseline top-K on the upper line and ours top-K on the lower line')
parser.add_argument('--min_base_hits', default=0, type=int,
                    help='category "better" only: minimum number of correct baseline matches in the top-K, so the '
                         'baseline is reasonably good and DFC-ReID is still shown to be better')
parser.add_argument('--exclude_json', action='append', default=[],
                    help='selection.json of earlier figures; their identities are not reused')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--test-batch', default=64, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--colors', default='rg', choices=['rg', 'cvd'],
                    help='rg: green/red borders; cvd: blue/orange (colorblind-safe)')
parser.add_argument('--out', default='figures/retrieval_vis', type=str, help='output path without extension')
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


def extract(net, img, label, modal):
    loader = data.DataLoader(TestData(img, label, transform=transform_test, img_size=(192, 384)),
                             batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, feat_fc = net(x, x, x, x, modal)
            feats.append(feat_fc.float().cpu())
    return torch.cat(feats).numpy()


def ranked(q_feat, g_feat, q_cams, g_cams):
    """Gallery indices sorted by similarity per query, with the eval_sysu camera rule applied."""
    order = np.argsort(-(q_feat @ g_feat.T), axis=1)
    out = []
    for qi in range(len(q_feat)):
        o = order[qi]
        if q_cams[qi] == 3:
            o = o[g_cams[o] != 2]
        out.append(o)
    return out


q_img, q_ids, q_cams = process_query_sysu(args.data_path, mode=args.mode)
g_img, g_ids, g_cams = process_gallery_sysu(args.data_path, mode=args.mode, trial=args.trial)

ranks = {}
for name, method, path in (('baseline', args.baseline_method, args.baseline),
                           ('ours', args.ours_method, args.ours)):
    net = build(method, path)
    qf = extract(net, q_img, q_ids, modal=2)
    gf = extract(net, g_img, g_ids, modal=1)
    ranks[name] = ranked(qf, gf, q_cams, g_cams)
    del net
    torch.cuda.empty_cache()

hit_b = np.array([g_ids[r[0]] == q_ids[i] for i, r in enumerate(ranks['baseline'])])
hit_o = np.array([g_ids[r[0]] == q_ids[i] for i, r in enumerate(ranks['ours'])])
pools = {
    'improved': np.where(~hit_b & hit_o)[0],
    'failure': np.where(~hit_b & ~hit_o)[0],
    'degraded': np.where(hit_b & ~hit_o)[0],
    'random': np.arange(len(q_img)),
}
hits = lambda name, q: int(sum(g_ids[g] == q_ids[q] for g in ranks[name][q][:args.topk]))
pools['better'] = np.array([q for q in range(len(q_img)) if hit_o[q] and
                            hits('baseline', q) >= args.min_base_hits and hits('ours', q) > hits('baseline', q)],
                           dtype=int)
counts = {k: int(len(v)) for k, v in pools.items() if k != 'random'}
counts['both_correct'] = int((hit_b & hit_o).sum())
counts['total'] = int(len(q_img))
print('Rank-1 categories over all queries (trial {}):'.format(args.trial), counts)

rng = random.Random(args.seed)
all_used = set()  # identities already shown in any figure
for jp in args.exclude_json:
    all_used |= {r['pid'] for r in json.load(open(jp))['rows']}


def select_rows():
    rows = []  # (category, query index)
    if args.queries:
        return [('manual', int(q)) for q in args.queries.split(',')]
    for spec in (args.rows or ['improved:4', 'failure:1']):
        cat, n = spec.split(':')
        cand = sorted(pools[cat].tolist())
        rng.shuffle(cand)  # seeded tie-break
        if args.curate:
            gain = lambda q: (sum(g_ids[g] == q_ids[q] for g in ranks['ours'][q][:args.topk]) -
                              sum(g_ids[g] == q_ids[q] for g in ranks['baseline'][q][:args.topk]))
            cand.sort(key=lambda q: -gain(q))
        pick = []
        for q in cand:  # different identity for every row
            if int(q_ids[q]) not in all_used and len(pick) < int(n):
                pick.append(q)
                all_used.add(int(q_ids[q]))
        rows += [(cat, q) for q in (pick if args.curate else sorted(pick))]
    return rows


figs = [select_rows() for _ in range(args.num_figs)]
rows = [r for f in figs for r in f]

# ---------------------------------------------------------------- drawing
OK, BAD = ('#1baf7a', '#e5484d') if args.colors == 'rg' else ('#2a78d6', '#eb6834')
K = args.topk
NUM_WORDS = {1: 'one', 3: 'three', 5: 'five', 10: 'ten', 20: 'twenty'}
CW, CH, GAP, QGAP = 1.0, 2.0, 0.12, 0.6        # cell width/height, gap between cells, gap around query
ncols_w = 2 * K * (CW + GAP) + CW + 2 * QGAP


def cell(ax, x, y, path, edge=None):
    im = Image.open(path).convert('RGB').resize((96, 192), Image.BILINEAR)
    ax.imshow(np.asarray(im), extent=(x, x + CW, y + CH, y), zorder=1)
    if edge:
        ax.add_patch(Rectangle((x, y), CW, CH, fill=False, ec=edge, lw=1.6, zorder=2))


def draw(sub_rows, out):
    n_main = sum(1 for c, _ in sub_rows if c != 'failure')
    sep = 0.5 if n_main < len(sub_rows) else 0.0
    H = len(sub_rows) * (CH + GAP) + sep + 0.7
    fig, ax = plt.subplots(figsize=(ncols_w * 0.42, H * 0.42))
    ax.set_xlim(0, ncols_w)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis('off')
    xq = K * (CW + GAP) + QGAP
    ax.text(xq - QGAP - K * (CW + GAP) / 2, 0.35, 'Baseline', ha='center', va='center', fontsize=8)
    ax.text(xq + CW / 2, 0.35, 'Query', ha='center', va='center', fontsize=8)
    ax.text(xq + CW + QGAP + K * (CW + GAP) / 2, 0.35, 'DFC-ReID (ours)', ha='center', va='center', fontsize=8)
    y = 0.7
    for r, (cat, qi) in enumerate(sub_rows):
        if cat == 'failure' and sep and (r == 0 or sub_rows[r - 1][0] != 'failure'):
            ax.plot([0, ncols_w], [y + sep / 2 - GAP / 2] * 2, ls=(0, (4, 3)), color='#5b5b66', lw=0.8)
            y += sep
        cell(ax, xq, y, q_img[qi])
        for k in range(K):
            gb = ranks['baseline'][qi][k]
            cell(ax, xq - QGAP - (k + 1) * (CW + GAP) + GAP, y, g_img[gb], OK if g_ids[gb] == q_ids[qi] else BAD)
            go = ranks['ours'][qi][k]
            cell(ax, xq + CW + QGAP + k * (CW + GAP), y, g_img[go], OK if g_ids[go] == q_ids[qi] else BAD)
        y += CH + GAP
    for ext in ('pdf', 'png'):
        fig.savefig('{}.{}'.format(out, ext), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def draw_stack(sub_rows, out):
    LW = 1.7                                   # width of the 'Baseline' / 'Ours' label column
    BH = 2 * CH + GAP                          # height of one block (two retrieval lines)
    W = CW + 2 * QGAP + LW + K * (CW + GAP)
    H = len(sub_rows) * (BH + 2 * GAP) + 0.7
    fig, ax = plt.subplots(figsize=(W * 0.42, H * 0.42))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis('off')
    xs = CW + QGAP                             # x of the dashed vertical separator
    x0 = xs + QGAP + LW                        # x of the rank-1 column
    ax.text(CW / 2, 0.35, 'Query', ha='center', va='center', fontsize=8)
    ax.text(x0 + K * (CW + GAP) / 2, 0.35, 'Top {} retrieved results'.format(NUM_WORDS.get(K, K)), ha='center', va='center', fontsize=8)
    y = 0.7
    for r, (cat, qi) in enumerate(sub_rows):
        if r:
            ax.plot([0, W], [y - GAP] * 2, ls=(0, (4, 3)), color='#5b5b66', lw=0.8)
        ax.plot([xs, xs], [y, y + BH], ls=(0, (4, 3)), color='#5b5b66', lw=0.8)
        cell(ax, 0, y + (BH - CH) / 2, q_img[qi])
        for line, (name, label) in enumerate((('baseline', 'Baseline'), ('ours', 'DFC-ReID\n(ours)'))):
            yl = y + line * (CH + GAP)
            ax.text(xs + QGAP + LW / 2 - 0.1, yl + CH / 2, label, ha='center', va='center', fontsize=8)
            for k in range(K):
                g = ranks[name][qi][k]
                cell(ax, x0 + k * (CW + GAP), yl, g_img[g], OK if g_ids[g] == q_ids[qi] else BAD)
        y += BH + 2 * GAP
    for ext in ('pdf', 'png'):
        fig.savefig('{}.{}'.format(out, ext), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if args.layout == 'stack':
    draw = draw_stack
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
if args.num_figs > 1:
    for n, f in enumerate(figs, 1):
        draw(f, '{}_{}'.format(args.out, n))
else:
    draw(rows, args.out)
selection = {
    'mode': args.mode, 'trial': args.trial, 'topk': K, 'seed': args.seed, 'counts': counts,
    'rows': [{'category': c, 'query_index': int(q), 'query_path': q_img[q], 'pid': int(q_ids[q]),
              'baseline_rank1_correct': bool(hit_b[q]), 'ours_rank1_correct': bool(hit_o[q])} for c, q in rows],
    'baseline': args.baseline, 'ours': args.ours,
}
with open(args.out + '_selection.json', 'w') as f:
    json.dump(selection, f, indent=2)
print('saved {}.pdf / .png and {}_selection.json'.format(args.out, args.out))
