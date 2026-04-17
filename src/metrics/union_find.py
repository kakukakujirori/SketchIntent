from collections import Counter


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.attr = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank.get(rx, 0) < self.rank.get(ry, 0):
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank.get(rx, 0) == self.rank.get(ry, 0):
            self.rank[rx] = self.rank.get(rx, 0) + 1
        if ry in self.attr:
            implied = self.attr[ry]
            if rx in self.attr:
                if self.attr[rx] != implied:
                    raise ValueError(f"Attribute conflict for {rx}: existing {self.attr[rx]}, implied {implied}")
            else:
                self.attr[rx] = implied
            del self.attr[ry]

    def get_attr(self, x):
        root = self.find(x)
        if root not in self.attr:
            return None
        return self.attr[root]

    def set_attr(self, x, value: float):
        root = self.find(x)
        if root in self.attr and self.attr[root] != value:
            raise ValueError(f"Attribute conflict for {root}: existing {self.attr[root]}, new {value}")
        self.attr[root] = value


class UnionFindWeighted:
    def __init__(self, mod: float | None = None, tol: float = 1e-4):
        self.parent = {}
        self.offset = {}
        self.rank = {}
        self.attr = {}
        self.mod = mod
        self.tol = tol

    def _mod(self, value: float | None):
        if value is None:
            return None
        return value % self.mod if self.mod is not None else value

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.offset[x] = 0

        # path to the root
        path = []
        node = x
        while self.parent[node] != node:
            path.append(node)
            node = self.parent[node]
        root = node

        # accumulate offsets from the root
        cum = 0
        cum_offsets = [0] * len(path)
        for i in range(len(path) - 1, -1, -1):
            cum = self._mod(self.offset[path[i]] + cum)
            cum_offsets[i] = cum

        # path compression
        for i, node in enumerate(path):
            self.parent[node] = root
            self.offset[node] = cum_offsets[i]

        return root, (cum_offsets[0] if path else 0)

    def union(self, x, y, diff: float = 0):
        """
        Union x and y, with the constraint that label(y) = label(x) + diff.
        """
        rx, ox = self.find(x)
        ry, oy = self.find(y)
        if rx == ry:
            if abs(self._mod(oy - ox) - self._mod(diff)) > self.tol:
                raise ValueError(f"Inconsistent: root {rx}, label({y})-label({x}) is {self._mod(oy-ox)}, expected {self._mod(diff)}")
        if self.rank.get(rx, 0) < self.rank.get(ry, 0):
            rx, ry = ry, rx
            ox, oy = oy, ox
            diff = -diff
        self.parent[ry] = rx
        self.offset[ry] = self._mod(ox + diff - oy)
        if self.rank.get(rx, 0) == self.rank.get(ry, 0):
            self.rank[rx] = self.rank.get(rx, 0) + 1
        if ry in self.attr:
            implied = self._mod(self.attr[ry] - self.offset[ry])
            if rx in self.attr:
                if abs(self._mod(self.attr[rx]) - implied) > self.tol:
                    raise ValueError(f"Attribute conflict for {rx}: existing {self.attr[rx]}, implied {implied}")
            else:
                self.attr[rx] = implied
            del self.attr[ry]

    def get_attr(self, x):
        root, offset = self.find(x)
        if root not in self.attr:
            return None
        return self._mod(self.attr[root] + offset)

    def set_attr(self, x, value: float):
        root, offset = self.find(x)
        implied_root_value = self._mod(value - offset)
        if root in self.attr and abs(self._mod(self.attr[root]) - implied_root_value) > self.tol:
            raise ValueError(f"Attribute conflict for {root}: existing {self.attr[root]}, new {implied_root_value}")
        self.attr[root] = implied_root_value


def BCubedF1(
    gt_uf: UnionFind,
    pred_uf: UnionFind,
    gt_items: list | None = None,
    pred_items: list | None = None,
) -> tuple[float, float, float]:
    """
    Compute B-cubed precision / recall / F1 between two UnionFind partitions.

    Parameters
    ----------
    gt_uf : UnionFind
        Ground-truth partition.
    pred_uf : UnionFind
        Predicted partition.
    gt_items : list[Any], optional
        Items for recall denominator. Defaults to gt_uf.parent.keys().
    pred_items : list[Any], optional
        Items for precision denominator. Defaults to pred_uf.parent.keys().

    Returns
    -------
    tuple[float, float, float]
        Precision, Recall, F1.
    """
    if gt_items is None:
        gt_items = list(gt_uf.parent.keys())
    if pred_items is None:
        pred_items = list(pred_uf.parent.keys())

    all_items = list(set(gt_items) | set(pred_items))
    if not all_items:
        return (0.0, 0.0, 0.0)

    # Root of each item in each partition
    gt_root = {x: gt_uf.find(x) for x in all_items}
    pred_root = {x: pred_uf.find(x) for x in all_items}

    # Cluster sizes
    gt_sizes = Counter(gt_root.values())
    pred_sizes = Counter(pred_root.values())

    # Intersection size for each (pred_cluster, gt_cluster) pair
    pair_sizes = Counter()
    for x in all_items:
        pair_sizes[(pred_root[x], gt_root[x])] += 1

    # Precision is micro-averaged over predicted items
    precision_sum = 0.0
    for x in pred_items:
        pr = pred_root[x]
        gr = gt_root[x]
        inter = pair_sizes[(pr, gr)]
        precision_sum += inter / pred_sizes[pr]

    # Recall is micro-averaged over ground-truth items
    recall_sum = 0.0
    for x in gt_items:
        pr = pred_root[x]
        gr = gt_root[x]
        inter = pair_sizes[(pr, gr)]
        recall_sum += inter / gt_sizes[gr]

    precision = precision_sum / len(pred_items) if pred_items else 0.0
    recall = recall_sum / len(gt_items) if gt_items else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1
