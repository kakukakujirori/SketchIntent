"""
Compute Weisfeiler-Lehman Graph Kernel
"""
import glob
import os
from typing import List, Dict, Tuple

import numpy as np

from bleu_check import load_sketch, align_sketch
from graph import SketchConstraintGraph


def _init_labels(g: SketchConstraintGraph) -> Dict[Tuple[str, int], str]:
    return dict(g.node_labels)  # ジオメトリタイプが初期ラベル

def _aggregate(node: Tuple[str, int],
               graph: SketchConstraintGraph,
               curr_labels: Dict[Tuple[str, int], str]) -> str:
    """
    ノードiの新ラベル = hash(自ラベル + sorted近傍情報)
    近傍情報 = (制約タイプ, 対向ノードの現ラベル) のペア
    """
    neighbors = []
    for e in graph.adj[node]:
        other_nodes = [r for r in e.refs if r != node]
        for n in other_nodes:
            neighbors.append(f"{e.ctype}:{curr_labels.get(n, 'UNK')}")

        # 自分自身しか参照していない制約（実質単項制約）
        if len(other_nodes) == 0:
            neighbors.append(f"{e.ctype}:unary")

    neighbor_str = '_'.join(sorted(neighbors))
    return f"{curr_labels[node]}|[{neighbor_str}]"

def node_aligned_wl_similarity(g1: SketchConstraintGraph,
                                g2: SketchConstraintGraph,
                                h: int = 3) -> float:
    """
    各ノードのラベル系列を h ステップ比較。
    ノード集合共有を活かした aligned 版。

    Returns similarity in [0,1].
    """
    labels1 = _init_labels(g1)
    labels2 = _init_labels(g2)
    nodes = list(set(labels1.keys()) | set(labels2.keys()))

    # ノードごとの一致ステップ数を記録
    node_match_counts = {n: int(labels1.get(n) == labels2.get(n)) for n in nodes}

    for step in range(h):
        new1 = {n: _aggregate(n, g1, labels1) if n in g1.node_labels else "MISSING" for n in nodes}
        new2 = {n: _aggregate(n, g2, labels2) if n in g2.node_labels else "MISSING" for n in nodes}
        for n in nodes:
            if new1[n] == new2[n]:
                node_match_counts[n] += 1
        labels1, labels2 = new1, new2

    # 各ノード: (一致ステップ数)/(全ステップ数) の平均
    total_steps = h + 1  # h=0 (初期) + h反復
    per_node = [node_match_counts[n] / total_steps for n in nodes]
    return float(np.mean(per_node)) if per_node else 1.0


def wl_histogram_similarity(g1: SketchConstraintGraph,
                             g2: SketchConstraintGraph,
                             h: int = 3) -> float:
    """
    標準WL: ラベルヒストグラムのコサイン類似度。
    node-aligned が使えない場合（ノード集合が異なるケース）の fallback。
    """
    from collections import Counter
    labels1 = _init_labels(g1)
    labels2 = _init_labels(g2)
    hist1, hist2 = Counter(labels1.values()), Counter(labels2.values())

    for _ in range(h):
        new1 = {n: _aggregate(n, g1, labels1) for n in g1.node_labels}
        new2 = {n: _aggregate(n, g2, labels2) for n in g2.node_labels}
        hist1.update(new1.values())
        hist2.update(new2.values())
        labels1, labels2 = new1, new2

    keys = set(hist1) | set(hist2)
    v1 = np.array([hist1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([hist2.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 0 else 1.0


def wl_score(pred: SketchConstraintGraph,
             gts: List[SketchConstraintGraph],
             h: int = 3) -> dict:
    sims = [node_aligned_wl_similarity(gt, pred, h) for gt in gts]
    best = int(np.argmax(sims))
    return {'score': sims[best], 'best_gt_idx': best, 'all_scores': sims}


if __name__ == '__main__':

    gt1_folder = "2_eko_old"
    gt2_folder = "2_mine_old"
    preservable_removed_folder = "2_mine_old_preservable_removed"
    auxiliary_removed_folder = "2_mine_old_auxiliary_removed"

    summary = {
        "preservable_removed_scores": [],
        "auxiliary_removed_scores": [],
        "preservable_removed_count": 0,
        "auxiliary_removed_count": 0,
        "tie_count": 0,
    }

    for filepath in sorted(glob.glob(os.path.join(gt1_folder, "*.FCStd"))):
        basename = os.path.basename(filepath)
        #print(f"{basename=}")

        # Load sketches
        gt1_path = os.path.join(gt1_folder, basename)
        gt2_path = os.path.join(gt2_folder, basename)
        preservable_removed_path = os.path.join(preservable_removed_folder, basename)
        auxiliary_removed_path = os.path.join(auxiliary_removed_folder, basename)

        sketch_base = load_sketch(gt1_path.replace(".FCStd", ".py"))
        sketch_gt1 = load_sketch(gt1_path)
        sketch_gt2 = load_sketch(gt2_path)
        sketch_preservable_removed = load_sketch(preservable_removed_path)
        sketch_auxiliary_removed = load_sketch(auxiliary_removed_path)

        sketch_gt1 = align_sketch(sketch_gt1, sketch_base.entities)
        sketch_gt2 = align_sketch(sketch_gt2, sketch_base.entities)
        sketch_preservable_removed = align_sketch(sketch_preservable_removed, sketch_base.entities)
        sketch_auxiliary_removed = align_sketch(sketch_auxiliary_removed, sketch_base.entities)

        result_preservable = wl_score(
            SketchConstraintGraph.from_sketch(sketch_preservable_removed),
            [SketchConstraintGraph.from_sketch(sketch_gt1),
             SketchConstraintGraph.from_sketch(sketch_gt2)],
        )
        result_auxiliary = wl_score(
            SketchConstraintGraph.from_sketch(sketch_auxiliary_removed),
            [SketchConstraintGraph.from_sketch(sketch_gt1),
             SketchConstraintGraph.from_sketch(sketch_gt2)],
        )

        summary["preservable_removed_scores"].append(result_preservable["score"])
        summary["auxiliary_removed_scores"].append(result_auxiliary["score"])

        if result_preservable["score"] > result_auxiliary["score"]:
            summary["preservable_removed_count"] += 1
        elif result_preservable["score"] < result_auxiliary["score"]:
            summary["auxiliary_removed_count"] += 1
        else:
            summary["tie_count"] += 1

        print(f"{basename}: preservable_removed={result_preservable['score']:.4f}  auxiliary_removed={result_auxiliary['score']:.4f}")

    print(f"\n=== Summary ===")
    total_counts = summary["preservable_removed_count"] + summary["auxiliary_removed_count"] + summary["tie_count"]
    print(f"Preservable removed counts: {summary['preservable_removed_count']}/{total_counts}")
    print(f"Auxiliary removed counts: {summary['auxiliary_removed_count']}/{total_counts}")
    print(f"Tie counts: {summary['tie_count']}/{total_counts}")
    print(f"Average Preservable removed score: {sum(summary['preservable_removed_scores']) / len(summary['preservable_removed_scores']):.4f}")
    print(f"Average Auxiliary removed score: {sum(summary['auxiliary_removed_scores']) / len(summary['auxiliary_removed_scores']):.4f}")