"""
graph.py  –  制約グラフのデータ構造

CEdge      : 制約1件を表す不変データクラス
SketchConstraintGraph : datalib.Sketch からグラフを構築するクラス

sketchgraphs の Constraint オブジェクトには複数の実装バリアントが
存在するため、_extract_entity_refs() で段階的フォールバックを行う。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# CEdge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CEdge:
    """
    制約1件を表す不変オブジェクト。

    ctype : 制約タイプ名 (str)。ConstraintType.name を使用。
    refs  : (entity_id, sub_entity_pos) ペアのソート済みタプル。
            entity_id   : str (UUID)
            sub_entity_pos : int
              FreeCAD PointPos 対応: 0=edge/none, 1=start, 2=end, 3=mid
    value : 寸法制約の数値 (非寸法制約では None)
    """
    ctype: str
    refs: Tuple[Tuple[str, int], ...]
    value: Optional[float] = None

    def attachment_key(self) -> Tuple[Tuple[str, int], ...]:
        """
        同一アタッチメント = 同一エンティティ集合 + 同一点位置の組み合わせ。
        nGED のグループ化キーとして使用。
        """
        return self.refs


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _get_entity_type_name(entity: Any) -> str:
    """
    entity の型名を返す。
    sketchgraphs Entity は .entityType (Enum) を持つ場合がある。
    """
    t = getattr(entity, 'entityType', None)
    if t is not None:
        return t.name if hasattr(t, 'name') else str(t)
    return type(entity).__name__


def _is_construction(entity: Any) -> bool:
    """構築ジオメトリかどうかを判定。"""
    return bool(
        getattr(entity, 'isConstruction', False)
        or getattr(entity, 'Construction', False)
    )


def _get_constraint_type_name(constraint: Any) -> str:
    """
    制約オブジェクトから型名 (str) を取得。
    sketchgraphs は .constraintType (Enum) を持つ。
    """
    t = (
        getattr(constraint, 'constraintType', None)
        or getattr(constraint, 'Type', None)
        or getattr(constraint, 'type', None)
    )
    if t is None:
        return 'Unknown'
    return t.name if hasattr(t, 'name') else str(t)


def _extract_entity_refs(constraint: Any) -> List[Tuple[Any, int]]:
    """
    制約オブジェクトから (entity_id, sub_entity_pos) のリストを取得。

    sketchgraphs Constraint の実装バリアントに応じた段階的フォールバック:

    パターン A  entityIds + subEntityIds が別属性
                例: c.entityIds = ('uuid-A', 'uuid-B')
                    c.subEntityIds = (1, 2)   # PointPos

    パターン B  entityIds のみ (サブエンティティ情報なし)
                例: c.entityIds = ('uuid-A', 'uuid-B')

    パターン C  references が [(eid, pos), ...] 形式

    パターン D  localPoints / referencePoints 形式
                LocalPoint オブジェクトが entity_id + point_id を持つ
    """

    # --- パターン A / B : entityIds ベース ---
    entity_ids = getattr(constraint, 'entityIds', None)
    if entity_ids is not None:
        entity_ids = list(entity_ids)
        sub_ids_raw = getattr(constraint, 'subEntityIds', None)

        if sub_ids_raw is not None:
            sub_ids = list(sub_ids_raw)
            # 長さ不足時はゼロ補完
            sub_ids += [0] * (len(entity_ids) - len(sub_ids))
        else:
            sub_ids = [0] * len(entity_ids)

        # sub_entity_pos は int / Enum どちらも許容
        result = []
        for eid, pos in zip(entity_ids, sub_ids):
            pos_int = pos.value if hasattr(pos, 'value') else int(pos)
            result.append((str(eid), pos_int))
        return result

    # --- パターン C : references リスト ---
    references = getattr(constraint, 'references', None)
    if references is not None:
        result = []
        for ref in references:
            if isinstance(ref, (tuple, list)) and len(ref) >= 2:
                pos = ref[1]
                pos_int = pos.value if hasattr(pos, 'value') else int(pos)
                result.append((str(ref[0]), pos_int))
            else:
                result.append((str(ref), 0))
        return result

    # --- パターン D : localPoints / LocalPoint オブジェクト ---
    local_points = (
        getattr(constraint, 'localPoints', None)
        or getattr(constraint, 'referencePoints', None)
    )
    if local_points is not None:
        result = []
        for lp in local_points:
            eid = getattr(lp, 'entity_id', getattr(lp, 'entityId', str(lp)))
            pos = getattr(lp, 'point_id', getattr(lp, 'pointId', 0))
            pos_int = pos.value if hasattr(pos, 'value') else int(pos)
            result.append((str(eid), pos_int))
        return result

    # --- パターン E : get_references() メソッド ---
    get_refs = getattr(constraint, 'get_references', None)
    if callable(get_refs):
        refs = get_refs()
        if refs:
            result = []
            for ref in refs:
                if isinstance(ref, str) and '.' in ref:
                    eid_str, pos_str = ref.split('.', 1)
                    pos_int = 0
                    if pos_str == 'start': pos_int = 1
                    elif pos_str == 'end': pos_int = 2
                    elif pos_str == 'mid': pos_int = 3
                    result.append((eid_str, pos_int))
                else:
                    result.append((str(ref), 0))
            return result

    return []


# ---------------------------------------------------------------------------
# SketchConstraintGraph
# ---------------------------------------------------------------------------

class SketchConstraintGraph:
    """
    datalib.Sketch から制約グラフを構築する。

    ノード : 非構築ジオメトリ  (entity_id -> ラベル文字列)
    エッジ : 制約              (CEdge のリスト)
    adj    : entity_id -> 隣接 CEdge リスト

    Parameters
    ----------
    sketch : datalib.Sketch
        sketchgraphs の Sketch オブジェクト。
        .entities (dict[str, Entity]) と .constraints (list[Constraint])
        を持つことを前提とする。
    """

    def __init__(self) -> None:
        self.node_labels: Dict[Tuple[str, int], str] = {}          # (entity_id, pos) -> type名
        self.edges: List[CEdge] = []
        self.adj: Dict[Tuple[str, int], List[CEdge]] = defaultdict(list)

    @classmethod
    def from_sketch(cls, sketch: Any) -> SketchConstraintGraph:
        g = cls()
        valid_eids: Set[str] = set()

        # ---- ノード構築 ----
        for eid, entity in sketch.entities.items():
            if _is_construction(entity):
                continue
            eid_str = str(eid)
            etype = _get_entity_type_name(entity)
            valid_eids.add(eid_str)
            
            base_node = (eid_str, 0)
            g.node_labels[base_node] = etype
            
            sub_positions = []
            if etype in ('Line', 'LineSegment'):
                sub_positions = [1, 2]
            elif etype in ('Circle',):
                sub_positions = [3]
            elif etype in ('Arc', 'ArcOfCircle'):
                sub_positions = [1, 2, 3]
                
            for pos in sub_positions:
                sub_node = (eid_str, pos)
                g.node_labels[sub_node] = "Point"
                edge_type = "HasPoint"
                if pos == 1: edge_type = "HasStart"
                elif pos == 2: edge_type = "HasEnd"
                elif pos == 3: edge_type = "HasCenter"
                
                s_edge = CEdge(edge_type, (base_node, sub_node), None)
                g.edges.append(s_edge)
                g.adj[base_node].append(s_edge)
                g.adj[sub_node].append(s_edge)

        # ---- エッジ構築 ----
        constraints = sketch.constraints.values() if isinstance(sketch.constraints, dict) else sketch.constraints
        for c in constraints:
            ctype = _get_constraint_type_name(c)
            raw_refs = _extract_entity_refs(c)

            # 非構築エンティティのみ
            filtered = [
                (eid, pos) for eid, pos in raw_refs if eid in valid_eids
            ]
            if not filtered:
                continue

            # canonical order
            filtered.sort()
            refs = tuple(filtered)
            
            # Ensure referenced sub-nodes exist
            for ref in refs:
                if ref not in g.node_labels:
                    g.node_labels[ref] = "Point"
                    base_node = (ref[0], 0)
                    s_edge = CEdge("HasPoint", (base_node, ref), None)
                    g.edges.append(s_edge)
                    g.adj[base_node].append(s_edge)
                    g.adj[ref].append(s_edge)

            value = getattr(c, 'value', None)
            if isinstance(value, float) and value == 0.0:
                value = None

            edge = CEdge(ctype, refs, value)
            g.edges.append(edge)
            for ref in filtered:
                g.adj[ref].append(edge)

        return g

    # ----------------------------------------------------------------
    # デバッグ用
    # ----------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Nodes : {len(self.node_labels)}",
            f"Edges : {len(self.edges)}",
        ]
        from collections import Counter
        ctype_count = Counter(e.ctype for e in self.edges)
        for ct, n in ctype_count.most_common():
            lines.append(f"  {ct:25s} x{n}")
        return "\n".join(lines)

    @staticmethod
    def diagnose_constraint(constraint: Any) -> str:
        """
        制約オブジェクトの属性を列挙して返す。
        sketchgraphs のバージョンが不明な場合のデバッグ用。
        """
        attrs = {
            k: getattr(constraint, k)
            for k in dir(constraint)
            if not k.startswith('_')
        }
        return str(attrs)