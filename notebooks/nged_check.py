# """
# nged.py  –  Normalized Graph Edit Distance による制約セット評価

# 公開 API
# --------
# compute_nged(gt, pred)  ->  float          # スケッチ1ペアの距離 [0,1]
# nged_score(pred, gts)   ->  dict           # GT リストに対するスコア
# NGEDAccumulator                            # バッチ集計 (ConstraintBLEU 対応)

# nGED の定義
# -----------
# 同一ノード集合 (共有プリミティブ) を前提とするため、
# グラフ編集距離はエッジの挿入・削除・置換コストのみに帰着する。

#     GED = Σ_{attachment_key} min_edit_cost(GT 側制約リスト, Pred 側制約リスト)

# 同一 attachment_key 内の複数制約は Hungarian アルゴリズムで最適マッチング。
# """

# from __future__ import annotations

# import glob
# import json
# import os
# from collections import defaultdict
# from typing import Dict, List, Optional

# import numpy as np
# from scipy.optimize import linear_sum_assignment

# from bleu_check import load_sketch, align_sketch
# from graph import CEdge, SketchConstraintGraph


# # ---------------------------------------------------------------------------
# # 制約タイプ置換コスト
# # ---------------------------------------------------------------------------

# # 同ファミリー内の置換は完全置換より安い (0.5)
# _FAMILIES: Dict[str, frozenset] = {
#     'orientation': frozenset({'Horizontal', 'Vertical', 'Parallel', 'Perpendicular'}),
#     'coincidence': frozenset({'Coincident', 'PointOnObject', 'Tangent'}),
#     'equality':    frozenset({'Equal', 'Radius', 'Diameter'}),
#     'dimension':   frozenset({'Distance', 'DistanceX', 'DistanceY', 'Angle'}),
#     'fix':         frozenset({'Fixed', 'Block'}),
#     'symmetry':    frozenset({'Symmetric'}),
# }
# _TYPE_TO_FAMILY: Dict[str, str] = {
#     t: f for f, ts in _FAMILIES.items() for t in ts
# }


# def _subst_cost(c1: str, c2: str) -> float:
#     if c1 == c2:
#         return 0.0
#     if _TYPE_TO_FAMILY.get(c1) == _TYPE_TO_FAMILY.get(c2):
#         return 0.5          # 同ファミリー内代替
#     return 1.0              # 完全置換


# # ---------------------------------------------------------------------------
# # attachment 単位の編集コスト
# # ---------------------------------------------------------------------------

# def _group_by_attachment(edges: List[CEdge]) -> Dict:
#     """attachment_key -> [ctype, ...] のマルチマップ"""
#     groups: Dict = defaultdict(list)
#     for e in edges:
#         groups[e.attachment_key()].append(e.ctype)
#     return groups


# def _attachment_edit_cost(gt_ctypes: List[str], pred_ctypes: List[str]) -> float:
#     """
#     同一 attachment 上の制約リスト間の最小編集コストを Hungarian 法で算出。

#     行列のサイズを max(n, m) に揃え、余剰行/列のコストを 1.0 (削除 or 挿入) で埋める。
#     """
#     n, m = len(gt_ctypes), len(pred_ctypes)
#     if n == 0:
#         return float(m)     # すべて挿入
#     if m == 0:
#         return float(n)     # すべて削除

#     size = max(n, m)
#     cost = np.ones((size, size), dtype=float)   # デフォルト: 削除 or 挿入
#     for i, ct_gt in enumerate(gt_ctypes):
#         for j, ct_pr in enumerate(pred_ctypes):
#             cost[i, j] = _subst_cost(ct_gt, ct_pr)

#     row_idx, col_idx = linear_sum_assignment(cost)
#     return float(cost[row_idx, col_idx].sum())


# # ---------------------------------------------------------------------------
# # スケッチ単位の nGED
# # ---------------------------------------------------------------------------

# def compute_nged(gt: SketchConstraintGraph, pred: SketchConstraintGraph) -> float:
#     """
#     GT と Pred の間の Normalized GED を返す [0, 1]。

#     0 = 完全一致, 1 = 最大相違 (全削除 + 全挿入に相当)

#     max_cost の定義:
#         len(gt.edges) + len(pred.edges)
#         ※ 全 GT 制約を削除し全 Pred 制約を挿入するコスト
#     """
#     gt_groups   = _group_by_attachment(gt.edges)
#     pred_groups = _group_by_attachment(pred.edges)
#     all_keys    = set(gt_groups) | set(pred_groups)

#     total_cost = 0.0
#     for key in all_keys:
#         gc = gt_groups.get(key, [])
#         pc = pred_groups.get(key, [])
#         total_cost += _attachment_edit_cost(gc, pc)

#     max_cost = len(gt.edges) + len(pred.edges)
#     return (total_cost / max_cost) if max_cost > 0 else 0.0


# def nged_score(pred: SketchConstraintGraph,
#                gts: List[SketchConstraintGraph]) -> Dict:
#     """
#     Pred と GT リストの間の nGED スコアを計算する。

#     BLEU と同様に「最も近い GT との距離」を採用する。

#     Returns
#     -------
#     {
#         'score'       : float,         # 1 - best_distance  (高いほど良い)
#         'distance'    : float,         # best nGED          (低いほど良い)
#         'best_gt_idx' : int,
#         'all_distances': [float, ...]
#     }
#     """
#     distances = [compute_nged(gt, pred) for gt in gts]
#     best = int(np.argmin(distances))
#     return {
#         'score':         1.0 - distances[best],
#         'distance':      distances[best],
#         'best_gt_idx':   best,
#         'all_distances': distances,
#     }


# # ---------------------------------------------------------------------------
# # バッチ集計クラス
# # ---------------------------------------------------------------------------

# class NGEDAccumulator:
#     """
#     複数スケッチにわたる nGED スコアを集計する。
#     ConstraintBLEU の __call__ / compute() インターフェースに準拠。

#     Usage
#     -----
#     acc = NGEDAccumulator()
#     for sketch in sketches:
#         acc(pred_sketch=sketch_pred, targets=[sketch_gt1, sketch_gt2])
#     result = acc.compute()
#     """

#     def __init__(self) -> None:
#         self._records: List[Dict] = []

#     def __call__(self,
#                  pred_sketch,
#                  targets: List,
#                  tag: Optional[str] = None,
#                  base_prims=None   # ConstraintBLEU との API 互換性用 (未使用)
#                  ) -> Dict:
#         """
#         スケッチ1件分のスコアを計算して蓄積する。

#         Parameters
#         ----------
#         pred_sketch : datalib.Sketch  予測スケッチ
#         targets     : list[datalib.Sketch]  GT スケッチリスト
#         tag         : str  識別用タグ (ファイル名等)
#         base_prims  : 無視 (ConstraintBLEU との互換引数)
#         """
#         pred_graph = SketchConstraintGraph.from_sketch(pred_sketch)
#         gt_graphs  = [SketchConstraintGraph.from_sketch(s) for s in targets]
#         result     = nged_score(pred_graph, gt_graphs)
#         if tag is not None:
#             result['tag'] = tag
#         self._records.append(result)
#         return result

#     def compute(self) -> Dict:
#         """
#         全蓄積スコアの集計統計を返す。

#         Returns
#         -------
#         {
#             'mean'       : float,
#             'median'     : float,
#             'std'        : float,
#             'n'          : int,
#             'per_sketch' : list[dict]
#         }
#         """
#         if not self._records:
#             return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'n': 0,
#                     'per_sketch': []}

#         scores = np.array([r['score'] for r in self._records], dtype=float)
#         return {
#             'mean':       float(scores.mean()),
#             'median':     float(np.median(scores)),
#             'std':        float(scores.std()),
#             'n':          len(self._records),
#             'per_sketch': self._records,
#         }

#     def reset(self) -> None:
#         """蓄積データをリセットする。"""
#         self._records.clear()


# # ---------------------------------------------------------------------------
# # __main__
# # ---------------------------------------------------------------------------

# if __name__ == '__main__':

#     gt1_folder                 = "2_eko_old"
#     gt2_folder                 = "2_mine_old"
#     preservable_removed_folder = "2_mine_old_preservable_removed"
#     auxiliary_removed_folder   = "2_mine_old_auxiliary_removed"

#     acc_preservable = NGEDAccumulator()
#     acc_auxiliary   = NGEDAccumulator()

#     for filepath in sorted(glob.glob(os.path.join(gt1_folder, "*.FCStd"))):
#         basename = os.path.basename(filepath)

#         # ---- パスの構築 ----
#         gt1_path                 = os.path.join(gt1_folder,                 basename)
#         gt2_path                 = os.path.join(gt2_folder,                 basename)
#         preservable_removed_path = os.path.join(preservable_removed_folder, basename)
#         auxiliary_removed_path   = os.path.join(auxiliary_removed_folder,   basename)

#         # ---- スケッチのロード ----
#         sketch_base                 = load_sketch(gt1_path.replace(".FCStd", ".py"))
#         sketch_gt1                  = load_sketch(gt1_path)
#         sketch_gt2                  = load_sketch(gt2_path)
#         sketch_preservable_removed  = load_sketch(preservable_removed_path)
#         sketch_auxiliary_removed    = load_sketch(auxiliary_removed_path)

#         # ---- アライメント (bleu_check.py と同手順) ----
#         sketch_gt1                 = align_sketch(sketch_gt1,                sketch_base.entities)
#         sketch_gt2                 = align_sketch(sketch_gt2,                sketch_base.entities)
#         sketch_preservable_removed = align_sketch(sketch_preservable_removed, sketch_base.entities)
#         sketch_auxiliary_removed   = align_sketch(sketch_auxiliary_removed,   sketch_base.entities)

#         # ---- nGED 計算・蓄積 ----
#         acc_preservable(
#             pred_sketch = sketch_preservable_removed,
#             targets     = [sketch_gt1, sketch_gt2],
#             tag         = basename,
#         )
#         acc_auxiliary(
#             pred_sketch = sketch_auxiliary_removed,
#             targets     = [sketch_gt1, sketch_gt2],
#             tag         = basename,
#         )

#     # ---- 集計結果の出力 ----
#     def _to_serializable(x):
#         """numpy スカラーを Python ネイティブ型に変換。"""
#         if isinstance(x, (np.floating, np.integer)):
#             return x.item()
#         if isinstance(x, dict):
#             return {k: _to_serializable(v) for k, v in x.items()}
#         if isinstance(x, list):
#             return [_to_serializable(v) for v in x]
#         return x

#     print("================")
#     print("\nPreservable Removed (nGED)")
#     print(json.dumps(_to_serializable(acc_preservable.compute()), indent=2))
#     print("\nAuxiliary Removed (nGED)")
#     print(json.dumps(_to_serializable(acc_auxiliary.compute()), indent=2))