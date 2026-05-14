# HistCAD-style OES に対する response-feasibility 指標の優位性

作成日: 2026-05-12

## Executive Summary

2D CAD sketch の constraint prediction では、同じ geometry を fully constrained にする
constraint set が複数存在する。評価指標は、単に「編集が成功するか」だけでなく、
人間が意図した設計関係を保つ parameterization になっているかを測る必要がある。

HistCAD-style OES は finite edit 後の success / preservation を測る強い baseline である。
しかし今回の source-conditioned 実験では、dimensional constraints を削った劣化 sketch を
OES が高く評価してしまう failure mode が確認された。具体的には、human heldout sketch から
dimensions を削るほど OES は `0.286 -> 1.000` と改善した。

これに対して、最終的な主指標は `d_feas` とする。`d_feas` は edit request の達成性と
relational design intent preservation を 1 つの first-order residual として測る。

結論:

> HistCAD OES は editability baseline として有用だが、design-intent-aware constraint
> inference の評価としては不十分である。`d_feas` は Vitruvion の relational intent mismatch
> を HistCAD OES より強く識別し、さらに HistCAD OES が逆方向に評価する dimensional
> dropout degradation を正しく penalize する。

したがって論文上の最も防御しやすい主張は、
「HistCAD を置き換える」ではなく、
「HistCAD-style editability では見えない design-intent / edit-feasibility の品質差を、
`d_feas` が補完し、この設定ではより適切に識別する」である。

## 背景: なぜ exact matching でも OES だけでも足りないか

CAD sketch の constraint set は一意ではない。例えば symmetry, midpoint, equal,
parallel, distance の組み合わせは、異なる constraint token sequence で同じ設計意図を
表し得る。そのため constraint command level の exact matching は過度に厳しい。

一方、HistCAD の Constraint-Aware Editability Benchmark は、この多様表現問題をかなりうまく回避している。[元論文](https://arxiv.org/html/2602.19171v2)では constraint command の一致ではなく、同じ target dimensional edit を適用した後に、指定された preservation set の constraints が保たれるかを見る。

HistCAD の benchmark instance は、実行可能な source CAD sequence、作成または変更される
target dimensional constraint、具体的な edit value と attachment site、そして edit 後に
満たされるべき preservation set からなる。target constraint は length, radius, diameter,
angle, distance を対象とし、edit は existing dimension の変更、または feasible attachment site
への new dimension の挿入である。preservation set は affected sketch 内の他の supported
constraints であり、source 側で edit 後も解けることが事前に validation される。

評価では、各 case に対して次の 3 つを判定する。

- `T`: target edit が満たされたか
- `V`: sketch validation と downstream rebuild に成功したか
- `P`: reachable case において preservation set の全 constraints が満たされたか

benchmark split の case 集合を `B`、`T` と `V` を満たす reachable cases を `R`、
さらに `P` も満たす successful cases を `S` とすると、HistCAD の 3 指標は次である。

```text
ER    = |R| / |B|
cPCSR = |S| / |R|
OES   = |S| / |B| = ER * cPCSR
```

つまり ER は「edit 後に valid CAD state へ到達できるか」、cPCSR は「到達できた case の中で
preservation set が保たれるか」、OES は「edit reachability と preservation の両方を満たす
strict overall rate」である。この設計は、design intent を finite redesign 後の constraint
preservation として測るという意味でかなり賢い。少なくとも、constraint の表現形式が違うだけで
不当に罰する exact matching よりは、本質的に良い baseline である。

しかしこの評価指標にはいくつかの問題点が存在する。今回の中心的な問題意識はここにある。

- HistCAD OES は強い baseline だが、まだ粗い。特に edit query や preservation set の選び方が metric の意味を大きく左右するにもかかわらず、その設計原理は十分には明示されない。
- また finite solve 後の success / failure を数える binary 指標なので、どの方向にどれだけ design intent から外れたのかを連続的に観測できない。
- さらに fully constrained に近く、design intent aware な sketch は、edit target と既存 dimensions の conflict によって reachable でなくなりやすい。一方、dimension を削った sketch は conflict が減って edit が通りやすくなる。その結果、fully constrained が通常望ましいとされるにもかかわらず、HistCAD OES は「より制約された良い sketch」を不当に低く見積もり、「dimension を落とした劣化 sketch」を高く見積もる可能性がある。

## 共通基盤: constraint ではなく primitive motion の接空間を見る

出発点は「正しい constraint set を一意に決めるのは難しい」という問題である。
Design intent aware な constraints の入れ方は多義的であり、同じ geometry と同じ意図を
Mirror で表すことも、Equal + Midpoint + Parallel で表すこともあり得る。

そこで constraint token や constraint type そのものを比較するのではなく、
その constraint set が primitive parameters `x` に許す動きの空間を見る。
直感的には、ある constraint set `C` は、元形状の近傍で「この sketch はどの方向に動けるか」
という可動多様体を primitive 空間の中に定める。Pred と GT が design intent として同等なら、
constraint の書き方が違っていても、この可動多様体、少なくともユーザが編集したときの
接線方向は近いはずである。

ただし、この可動多様体そのものは高次元空間に埋め込まれた非線形集合であり、
直接比較するのは難しい。そこで元形状のまわりで線形化し、接空間として比較する。
この線形化が以下で局所摂動を導入する理由である。

sketch `C` の primitive parameters を `x`、native dimensional constraint values を `p_C`
とする。dimensional constraints は

```text
h_j(x) - p_j = 0
```

relational constraints は

```text
r_k(x) = 0
```

として扱う。全 constraints をまとめると `F(x, p_C) = 0` と書け、これを線形化すると、

```text
J_x dx + J_p dp = 0
```

である。ここで `J_p` に列を持つのは dimensional constraints だけである。
native dimensions `p_C` を微小に動かしたとき、primitives `x` がどの方向に動けるかは

```text
R_C := dx / dp_C = - pinv(J_x) J_p （ここで pinv は擬逆行列）
```

で表される。これは constraint set `C` が持つ native dimensional values から primitive motion の
接空間への写像である。

しかし `p_C` の軸は sketch ごとに違う。Pred と GT で dimensional constraints の本数も意味も
異なるため、`R_C` をそのまま比較しても意味がない。そこで評価側で共通の edit request `q(x)` を導入する。
より正確には `q(x)` はスケッチの特定部分の dimension を指し、これ現在の値 `q(x_0)` から摂動変化させることを考える。
例えば `q(x) = Distance(A(x), B(x))` で「この距離を少し伸ばす」もしくは `q(x) = Radius(Arc(x))` で「この radius を少し変える」というような、どの sketch にも同じ意味で問い合わせられる probe である。

`q` の Jacobian を `dq = J_q dx` で表すと、`dq` を実現する native parameter change は

```text
J_q R_C dp_C = dq
dp_C = pinv(J_q R_C) dq
```

であり、すなわち共通 edit request `dq` に対する primitive response `dx = S_C^(q) dq` は

```text
dx = R_C dp_C = R_C pinv(J_q R_C) dq
S_C^(q) = R_C pinv(J_q R_C)
```

である。

`S_C^(q)` は、sketch `C` が共通 edit `q` に対して native dimensions の動きを最小限にしたときにどう反応するかを表す。
つまり、非線形な可動多様体全体を比較する代わりに、共通 edit 方向に沿った接ベクトルを比較する。


この構成から見ると、良い prediction が満たすべき条件は 2 つある。
1. `q(x)` の摂動が本当に実現可能であること
2. `q(x)` の摂動に対して source GT の relational constraints が保存されること

１つ目が数式でどう表されるかを確認する。
`S_C^(q)` は requested edit `dq` に対するprimitive motion なので、`dx = S_C^(q) dq` である。一方、実際に達成される edit 量は
`dq_achieved = J_q dx` である。したがって

```text
dq_achieved = J_q S_C^(q) dq
```

となる。任意の小さい `dq` に対して `dq_achieved = dq` となることが一次近似での edit 達成性なので、
行列として `J_q S_C^(q) = I` が必要になる。

２つ目に関しては `R` を source GT の relational constraints として、上記の議論と同様に `dR_achieved = J_R dx` を考えれば良い。

以上をまとめて、次の２式が満たされて欲しい条件である。

```text
J_q S_C^(q) = I          edit request が達成される
J_R S_C^(q) = 0          source GT の relational constraints が一次的に保存される
```

HistCAD は finite solve 後にこれらを binary に近い形で見る。一方、我々は元形状まわりの
接空間 response 上で、これらの条件への residual を直接測る。これが `d_feas` の考え方である。この考え方のメリットは２つある：
1. まず `q(x)` が target sketch に含まれているかどうか、さらには `q(x)` の摂動が target sketch を overconstrained ないし conflicting にするかに関わらず、上式を計算できる。HistCADの場合は `q(x)` のedit queryが target sketch にて conflicting を起こした場合は、たとえ target sketch が source sketch と意図的に等価であってもペナルティを受けてしまうが、一方我々の指標は局所変形の過程で `p_C` の連動的な摂動を許すため、それが発生しない。
2. さらには上式の差分は連続値のエラーとして観測可能である。HistCADの場合は edit が成功するかしないかのバイナリでしか評価が出来ないため、「どの程度」 source sketch が target に近いかは定量化出来ない。

## `d_feas`: edit feasibility と intent preservation を同時に測る

前節の定式化に基づき、 Pred の response `S_P^(q)` に対して、2 つの residual を定義する。

```text
T_P^(q) = J_q S_P^(q) - I_q       gain residual
L_P^(q) = J_{R_G} S_P^(q)         relation leak
```

`d_feas` はこれらを row-balanced に合成する。

```text
d_feas^(q)(P; G)
  = sqrt( ||T_P^(q)||_F^2 / k_q  +  ||L_P^(q)||_F^2 / n_R )

d_feas(P, G; R_G, Q_G)
  = |Q_G|^-1 Σ_{q ∈ Q_G} d_feas^(q)(P; G)
```

ここで `k_q` は edit probe `q` の residual row 数、`n_R` は source relation residual `R_G` の row 数である。
`Q_G` は source GT `G` から作った validated edit probes の集合であり、今回の実験ではその単純平均を使った。
したがって `d_feas` は「edit を達成できない」ことと「edit はできるが relation を壊す」ことを同じ枠組みで penalize する。

期待される挙動:

| case | gain residual `T` | relation leak `L` | `d_feas` |
|---|---:|---:|---:|
| source GT / intent-equivalent human | 0 | 0 | 0 |
| relation を壊す model output | 0 | >0 | >0 |
| dimension dropout | >0 | often 0 | >0 |
| 100% dropout | 1 | 0 | 1 |

per-relation normalized variant として `d_feas_tilde` も実装した。
これは各 relation block を GT 側 Jacobian Frobenius norm で割り、relation type の scale 差に
少し強くする variant である。実験上、結論は raw `d_feas` とほぼ同じだった。

## 実験設定

主実験は source-conditioned multi-reference setting で行った。
これは、複数 human reference を最初に 1 つの GT に統合するのではなく、
各 reference `G_i` をそれぞれ独立した source design intent として扱う設定である。
各 source は自分自身の edit probes `Q_i` と preserved relations `R_i` を定義し、
Pred は source ごとに `d_feas(P, G_i)` で評価される。最後に、Pred がどれか 1 つの
human design intent に十分近ければよいという multi-reference evaluation として、
low-is-good metrics では `min_i`、high-is-good metrics では `max_i` を取る。

対象:

- human annotations: batch 0 と batch 9 の計 47 common files
  - batch 0: Johnny / Sagar / Ryotaro
  - batch 9: Eko / Sagar / Ryotaro
- 各 file で 2 人を reference、残り 1 人を heldout とする rotation を作る。
- 有効な heldout rotations: 135
- Vitruvion model output が loaded かつ条件を満たす paired rotations: 114

各 source GT `G_i` から edit probes と preserved relations を作る。

```text
Q_i = G_i に含まれる dimensional constraints
R_i = G_i に含まれる relational constraints
```

`source_validated=True` のフラグ指定により `G_i` 自身で finite edit が成功するかを確認し、成功する probes だけを残す。この source validation は HistCAD benchmark の feasible edit site に合わせるためである。（現在の設定では常に成功するだろうが、念の為に実装している。）

評価対象:

```text
heldout_drop(r) = heldout human sketch から r% の dimensional constraints を削除
Vitruvion       = model prediction
```

multi-reference aggregation:

- `d_feas`: low-is-good なので `min_i`
- HistCAD OES: high-is-good なので `max_i`

主要実装と artifacts:

- `notebooks/exp_constrained_response_metric.py`:

  `d_resp`, `d_cresp`, `d_leak`, `d_feas`, `d_feas_tilde`, `d_align` を
  source-conditioned multi-reference setting で計算する主実験スクリプト。
  Vitruvion vs heldout-drop0 と dimensional dropout sweep の raw scores を出す。
- `notebooks/exp_source_conditioned_dropout.py`:

  同じ source-conditioned probes を使って HistCAD-style ER / cPCSR / OES と
  既存 response baseline を計算するスクリプト。
- `notebooks/analyze_dalign_evidence.py`:

  `d_feas` と HistCAD OES の paired Wilcoxon test、dropout trajectory、
  component median を集計する evidence aggregation。
- `notebooks/analyze_dfeas_trajectory.py`:

  dropout 率に対する `d_feas`, gain, leak の trajectory を詳しく見る補助解析。
- `notebooks/analyze_dfeas_outliers.py`:

  leak mean を壊す heavy-tail outlier を特定し、外れ値除外後の trajectory を確認する補助解析。
- `artifacts/constrained_response_scores_full.csv`:

  `drop_rate=0,100` と Vitruvion を含む `d_feas` 系 summary scores。
  実験 1 の Vitruvion vs heldout-drop0 に使う。
- `artifacts/constrained_response_scores_dropout.csv`:

  `drop_rate=0,25,50,75,100` の `d_feas` 系 summary scores。
  実験 2 と outlier 解析に使う。
- `artifacts/source_conditioned_dropout_scores.csv`:

  同じ rotations / probes に対する HistCAD-style ER / cPCSR / OES scores。
  OES baseline と dropout trajectory の比較に使う。

## 実験 1: Vitruvion vs human heldout の識別

まず、model prediction である Vitruvion と human heldout drop 0% を同じ rotation 上で比較した。
目的は、定性的に relation intent を壊しやすい Vitruvion を human より悪く評価できるかである。

validated, raw min aggregation, n=114:

`human-better`, `Vit-better`, `tie` は paired rotation ごとの勝敗数である。
距離指標では `Vitruvion - heldout` が正なら human-better、HistCAD OES では
`heldout - Vitruvion` が正なら human-better と数える。`z` は Wilcoxon signed-rank test の
normal approximation による z-score、`p` は two-sided p-value である。
tie は `|diff| <= 1e-9` とした。

| metric | human-better | Vit-better | tie | z | p |
|---|---:|---:|---:|---:|---:|
| HistCAD OES | 53 | 30 | 31 | 2.75 | 6.0e-3 |
| `d_feas` | 57 | 8 | 49 | 5.76 | 8.4e-9 |
| `d_feas_tilde` | 57 | 8 | 49 | 5.67 | 1.4e-8 |

解釈:

- HistCAD OES も一定の識別力は持つが、`d_feas` の方がはるかに強い。
- `d_feas` は Vitruvion の relational intent mismatch を leak 成分で検出しつつ、
  後述の dimensional dropout degradation も gain 成分で扱える。
- `tie` が多いが、これらの Vitruvion の結果が確かに human 並みであることは目視確認済み。

代表的な定性例として、`0/56013_752d02cb_0001_sketch00.FCStd` の
`Sagar+Ryotaro -> Johnny` rotation を見る。この case では HistCAD OES は heldout human Johnny より Vitruvion を高く評価するが、`d_feas` は Vitruvion を強く罰する。実際OES は finite edit が通ることを高く評価する一方、`d_feas` は Vitruvion の response が Sagar/Ryotaro source relations を一次的に壊すことを leak として検出する。

| pred | HistCAD OES ↑ | `d_feas` ↓ | gain | leak |
|---|---:|---:|---:|---:|
| heldout Johnny | 0.000 | ~0 | ~0 | ~0 |
| Vitruvion | 0.529 | 2.376 | ~0 | 2.552 |

[Heldout Human]

<img src="../artifacts/0_56013_752d02cb_0001_sketch00_Johnny.png" width="300" alt="Johnny heldout human">

[Vitruvion]

<img src="../artifacts/0_56013_752d02cb_0001_sketch00_Vitruvion.png" width="300" alt="Vitruvion prediction">

成分を見ると、Vitruvion は edit gain ではなく relation leak で罰されている。

| pred | median `min_d_feas` | median gain | median leak | median leak_tilde |
|---|---:|---:|---:|---:|
| Vitruvion | 0.0295 | ~0 | 0.0558 | 0.0306 |
| heldout-drop0 | ~0 | ~0 | ~0 | ~0 |

つまり Vitruvion は edit 自体は達成できるが、その response が source GT の relations を一次的に壊す方向になっている。

## 実験 2: dimensional dropout ablation

次に、human heldout sketch から dimensional constraints を `0,25,50,75,100%` 削除した。
この操作は relational constraints を残したまま dimensional parameter directions を減らすので、
design intent aware な fully constrained sketch から見れば明確な劣化である。

HistCAD OES はこの劣化を逆方向に評価した。

validated median:

| drop | `d_feas` | `d_feas_tilde` | gain | leak | HistCAD OES | ER | cPCSR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2857 | 0.2429 | 0.5000 |
| 25% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.8873 |
| 50% | 0.1429 | 0.1254 | 0.1667 | 0.0000 | 0.6944 | 0.6750 | 0.9167 |
| 75% | 0.4234 | 0.4000 | 0.3750 | 0.0000 | 0.8750 | 0.9000 | 0.9310 |
| 100% | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9310 |

結論:

- HistCAD OES は dropout 率が上がるほど改善する。
- 主因は ER の改善である。dimensions を削ると edit target と既存 dimensions が conflict しにくくなる。
- `d_feas` は gain residual によって 50% 以降で明確に悪化し、100% dropout で定義通り 1 になる。

Note: dimensional constraintsを抜くほど、スケッチ全体の可動自由度が上がって gain は一見低下（edit 達成率は上昇）しそうだが、これは間違い。`d_feas` の `S_P^(q)` は Pred が持つ「既存 native dimensional values を」微小に動かしたときの response だけを見るから、dimensional constraints を drop するほど edit 達成率は下がる。実際 100% dropout では `R_P` に dimensional columns がなくなるため `S_P^(q)=0`、したがって `J_q S_P^(q)-I=-I` となり、row-balanced gain は 1 になる。
すなわち gain はただの editability を測るのではなく、与えられた dimensional constraints が driving parameter としての役目を果たしているかまで見ていると言える。

within-rotation monotonicity でも、dropout が大きいほど `d_feas` は上がりやすい。
これは同じ rotation_id の中で隣り合う dropout rates を比較し、
`metric(higher drop) > metric(lower drop)` になった rotation の割合である。
median trajectory だけでなく、個別 rotation レベルでも劣化方向に動いているかを見るための診断である。

| transition | `d_feas` up | `d_feas_tilde` up |
|---|---:|---:|
| 0 -> 25% | 35.6% | 35.6% |
| 25 -> 50% | 46.7% | 48.1% |
| 50 -> 75% | 60.7% | 59.3% |
| 75 -> 100% | 83.0% | 85.2% |

これは HistCAD OES の failure mode と正反対の挙動である。

## 実験 3: heavy-tail 外れ値の確認

leak の mean は一部 dropout rate で巨大になった。これが主張を壊すか確認した。

`min_d_feas` は `d_feas` を 2 つの source GT のうち小さい方で集約した主指標である。

結果:

| criterion | unique rotations | rows |
|---|---:|---:|
| `min_d_feas > 10` | 1 | `9/55749...` at 25%, 75% |
| `median leak > 100` | 2 | `9/55749...`, `0/42540...` |

外れ値の内訳:

- `9/55749... Eko+Ryotaro -> Sagar`
  - 25%, 75% の両方で `min_d_feas` が `2.9e5 / 2.8e5` まで跳ねる。
  - `d_resp` も `1e6` scale まで跳ねており、`d_feas` 固有のバグではなく
    dropout 後の response が極端に ill-conditioned になったケース。
- `0/42540... Johnny+Sagar -> Ryotaro`
  - Sagar source だけ leak が巨大。
  - Johnny source では `d_feas=1.54` なので `min_i` aggregation では本体を大きく壊さない。

外れ値 2 rotations を除外した `min_d_feas` trajectory:

| drop | mean | p50 | p75 | p90 | p95 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.0395 | ~0 | ~0 | 0.0199 | 0.1115 | 2.0084 |
| 25% | 0.0831 | ~0 | 0.1667 | 0.2583 | 0.3507 | 0.5012 |
| 50% | 0.2078 | 0.1429 | 0.3750 | 0.5000 | 0.5910 | 1.5105 |
| 75% | 0.4852 | 0.4000 | 0.6000 | 1.0000 | 1.0000 | 8.4733 |
| 100% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

したがって、`d_feas` の median / quantile trajectory は外れ値に依存しない。
論文では mean ではなく median, quantile, paired counts を使うのが安全である。

## ここから導かれる結論

今回の実験から有用と分かった事実は 3 つである。

1. **HistCAD OES は editability baseline として有用だが、design intent evaluation としては不十分。**
   Dimensional dropout によって OES が改善するため、fully constrained / dimensioned sketch を
   不当に penalize する failure mode がある。

2. **`d_feas` は intent preservation と edit feasibility を単一 residual で扱える。**
   Vitruvion は leak 成分で罰され、dimensional dropout は gain 成分で罰される。
   HistCAD OES が dropout を逆方向に評価するのに対し、`d_feas` は正しい方向に悪化する。

3. **最終的な論文上の立て付けは `d_feas` vs HistCAD OES が最も真っ直ぐである。**
   HistCAD OES は比較 baseline として残し、`d_feas` を main metric にする。

主張の強さは次のように整理できる。

| claim | strength | evidence |
|---|---|---|
| HistCAD OES does not measure response-equivalence | strong | OES=1.0 でも response が異なる human-human case |
| HistCAD OES rewards dimensional dropout | strong | OES `0.286 -> 1.000` |
| `d_feas` detects Vitruvion relational intent mismatch better than OES | strong | p `8.4e-9` vs OES p `6.0e-3` |
| `d_feas` detects dimensional dropout degradation | strong | median `0 -> 1`, gain-driven |

推奨する final wording:

> HistCAD-style OES is a useful editability score, but it is not sufficient for
> evaluating design-intent-aware 2D sketch constraint inference. Our response-based
> metric directly tests the first-order conditions that a useful editable sketch should satisfy:
> the requested edit should be attainable, and the source design relations should remain invariant.
> On Vitruvion predictions, `d_feas` distinguishes human-like constraints more strongly than OES;
> under dimensional dropout, OES improves while `d_feas` worsens, exposing a concrete failure mode
> of HistCAD-style scoring.

## Appendix: relation-only diagnostic `d_align`

`d_align` は `d_feas` の relation leak 成分だけに注目した diagnostic である。

```text
d_align(P, G; R_G)
  = |Q_G|^-1 Σ_{q ∈ Q_G} ||J_{R_G} S_P^(q)|| / ||S_P^(q)||
```

これは Vitruvion の relational intent mismatch を非常に強く識別した
(`human-better=60`, `Vit-better=5`, `p=2.8e-10`)。
一方で edit attainability を見ないため、dimensional dropout は penalize しない。
したがって main metric ではない。`d_feas` の leak term が design-intent signal を持つことを示す補助診断として扱えるが、ここでは使用しない。

## Appendix: 途中で棄却した試み

### `d_resp`

`d_resp` は本研究の導出上は重要である。共通 edit request に対する response
`S_C^(q)` を作る発想は、最終的な `d_feas` の土台になっている。

しかし metric としては、geometry observable `J_phi S_C^(q)` だけを見るため、
Vitruvion のように dimensional constraints を多用する model output を良く見せてしまうことがある。
実際、Vitruvion vs human heldout では p=0.495 で、統計的に区別できなかった。

したがって `d_resp` は final metric ではなく、response-map 導出の背景として残す。

### `d_inv`

`d_inv` は local relation sensitivity を測る試みだった。

```text
s_C(r, q) = ||J_r S_C^(q)||
```

これは relation preservation の一次情報を入れるという方向性を示した点で重要だった。
しかし HistCAD cPCSR は finite solve 後の residual tolerance を見るため、
FreeCAD solve roundtrip や branch change による finite effect は `d_inv` では拾えない。

この経験から、最終的には relation term を response residual として直接入れる
`d_feas` に進んだ。relation-only view は補助診断としてのみ残す。

### `d_aug_beta`

`d_aug_beta` は geometry block と relation block を結合する試みだった。

```text
M_aug_C^(q; beta)
  = [ J_phi S_C^(q) / sqrt(n_phi);
      beta J_R S_C^(q) / sqrt(n_R) ]
```

これは Vitruvion の relation leakage を拾えることを示したが、`beta` に依存する。
また 100% dropout では `S_P=0` になり、geometry block も relation block も 0 へ collapse するため、
edit infeasibility を罰せない。

この失敗が、`S_P` と `S_G` の差を測るのではなく、`S_P` が満たすべき方程式
`J_q S=I`, `J_R S=0` の residual を直接測る `d_feas` につながった。

### constrained response (`d_cresp`)

`d_cresp` は pred response solver に source GT relations を hard constraint として入れる案だった。

```text
S_P^(q; R_G)
  = R_P pinv([J_q R_P; J_{R_G} R_P]) [I; 0]
```

しかし Vitruvion は dimensional parameter 数が多く、追加制約を満たす response が
見つかってしまうため、むしろ Vitruvion が良く見える方向に動いた。
paired comparison でも反対方向に有意であり、不採用とした。

### naive `S_C^(q)` match (`d_smatch`)

`d_feas` は `J_q S_P - I` と `J_{R_G} S_P` という方程式 residual を測る。素朴には、
そうした residual を取らずに **`S_P^(q)` と `S_G^(q)` を直接比較する** ほうが分かりやすい:

```text
d_smatch_rms^(q)(P, G) = ||S_P^(q) - S_G^(q)||_F / sqrt(n_x · k_q)
d_smatch_rel^(q)(P, G) = ||S_P^(q) - S_G^(q)||_F / max(||S_G^(q)||_F, eps)
d_smatch_sym^(q)(P, G) = 2 ||S_P^(q) - S_G^(q)||_F / (||S_P^(q)||_F + ||S_G^(q)||_F + eps)
```

各 probe について計算し、source ごとに平均、multi-reference では `min_i` を取る (低いほど良い)。
`d_feas` と同じ source-conditioned pipeline で 47 files × 3 rotations × 5 dropout rates と
Vitruvion を評価した (`notebooks/exp_smatch_metric.py`)。

**結果は 3 variants とも不合格**:

dropout trajectory (validated median, n=135):

| drop | `d_smatch_rms` | `d_smatch_rel` | `d_smatch_sym` | `d_feas` |
|---:|---:|---:|---:|---:|
| 0% | 0.0804 | 0.4327 | 0.4270 | 0.0000 |
| 25% | 0.1604 | 0.8096 | 0.7783 | 0.0000 |
| 50% | 0.1961 | 1.0188 | 1.0883 | 0.1667 |
| 75% | 0.2903 | 1.2139 | 1.4170 | 0.5000 |
| 100% | 0.1830 | 1.0000 | 2.0000 | 1.0000 |

Vitruvion vs heldout drop 0% paired (validated, 同 paired rotations n=79):

| metric | human-better | Vit-better | tie | z | p |
|---|---:|---:|---:|---:|---:|
| `d_smatch_rms` | 31 | 37 | 11 | -0.53 | 0.60 |
| `d_smatch_rel` | 33 | 35 | 11 | -0.79 | 0.43 |
| `d_smatch_sym` | 35 | 33 | 11 | 0.59 | 0.56 |
| `d_feas` | 40 | 6 | 33 | 4.06 | 4.9e-5 |

問題は 3 点。

1. **drop=0% で 0 にならない** (median 0.08–0.43)。同じ geometry の別 human annotation を penalize しており、design-intent ambiguity を吸収できない。
2. **dropout に対し非単調**: `d_smatch_rms`/`d_smatch_rel` は 75% でピーク、100% で逆戻り。100% dropout では `S_P=0` なので `||S_P - S_G||=||S_G||` に collapse し、75% dropout で誤った方向に動く `S_P` のほうが diff が大きくなる。within-rotation でも 75→100% は 80% の rotation で値が下がる。
3. **Vitruvion vs human の paired signal がない** (p > 0.4, しかも 2 variants では Vit がやや有利)。`d_feas` は同じ paired rotations で p=5e-5。

なぜ機能しないか — `S_C^(q) = R_C pinv(J_q R_C)` の `pinv` は **constraint set `C` 固有の native dim 空間 `p_C` で** 最小ノルム解を取る。`p_C` の基底は constraint の選び方に依存するので、intent equivalent な C, C' でも min-norm が指す方向が異なり、specific な `S_C^(q)` は一致しない。`d_feas` の `J_q S_P = I`, `J_{R_G} S_P = 0` は **parameterization 不変** な条件なので、この任意性に影響されない。

#### 最小例: 3 共線点での chain vs fan parameterization

primitive `x = (x_A, x_B, x_C)`, A=0, B=d₁, C=d₁+d₂ を 1 次元上に置く。両者とも fully constrained で同じ geometry を表す:

- GT (chain): `Distance(A, B)=d₁`, `Distance(B, C)=d₂`
- Pred (fan): `Distance(A, B)=d₁`, `Distance(A, C)=d₃`

probe `q = Distance(B, C)`, `J_q = (0, -1, 1)`. (Coincident は省略。relations 空、よって leak は trivially 0。)

```text
            [[-2, -1],                    [[-1, -1],
R_G = (1/3) [ 1, -1],          R_P = (1/3) [ 2, -1],
            [ 1,  2]]                     [-1,  2]]

J_q R_G = (0, 1)              J_q R_P = (-1, 1)
pinv((0, 1)) = (0, 1)^T       pinv((-1, 1)) = (-1/2, 1/2)^T

S_G = R_G (0, 1)^T            S_P = R_P (-1/2, 1/2)^T
    = (1/3)(-1, -1, 2)^T          = (0, -1/2, 1/2)^T
```

両方とも `J_q S = 1` (gain=0) で leak=0、すなわち `d_feas = 0`。
しかし `||S_G - S_P|| = 1/√6 ≈ 0.408`、`d_smatch > 0`。

意味: GT 側は BC が **native dim そのもの** なので `J_q R_G` が 1-hot `(0, 1)` になり、min-norm は「d₂ だけ動かす → C だけが動く」を選ぶ。Pred 側は BC が dim でなく d₁ と d₃ の両方が BC に効くので `J_q R_P = (-1, 1)`、min-norm は「d₁ を縮め d₃ を伸ばす → B と C を等量逆向きに動かす」を選ぶ。`p_C` 空間の基底が違うため、同じ probe に対する最小ノルム接ベクトルが異なる。これが Pseudo-inverse の段階で parameterization の違いが S_C^(q) に漏れ出る数値的中身である。

CAD annotation の現場でも、人によって距離を chain でつなぐか参照点からの fan で取るかは分かれる、よくある違いである。`d_feas` はこれを吸収するが、`d_smatch` は本物の design intent 違反と区別できずに罰してしまう。

Artifacts:

- `artifacts/smatch_vs_dfeas_summary.csv`
- `artifacts/smatch_vs_dfeas_details.csv`

## Appendix: 追加検証 2/3/4

### 2. positive / negative control

`notebooks/exp_v3_controls_and_robustness.py` で、同じ source-conditioned multi-reference
setting のまま control variants を追加した。

評価した control:

- `source_self`: source GT 自身を同じ source GT に対して評価する positive control
- `heldout`: 別 human の heldout sketch。表現は違っても design intent が近い positive control
- `dim_drop50/100`: heldout human から dimensional constraints を削除する negative control
- `rel_drop50/100`: heldout human から relational constraints を削除する negative control
- `Vitruvion`: model prediction

validated, `eps=1e-9`, multi-reference `min_i d_feas`:

| variant | n | median `d_feas` | p75 `d_feas` | median gain | median leak |
|---|---:|---:|---:|---:|---:|
| source self | 135 | ~0 | ~0 | ~0 | ~0 |
| human heldout | 135 | ~0 | ~0 | ~0 | ~0 |
| Vitruvion | 114 | 0.0295 | 0.1102 | ~0 | 0.0558 |
| dimension drop 50% | 135 | 0.1667 | 0.3709 | 0.1667 | ~0 |
| dimension drop 100% | 135 | 1.0000 | 1.0000 | 1.0000 | 0 |
| relation drop 50% | 135 | 0.4479 | 0.9020 | ~0 | 0.6737 |
| relation drop 100% | 135 | 0.5529 | 0.7819 | 0.3750 | 0.1988 |

解釈:

- `source_self` と `heldout` はほぼ 0 であり、同じ geometry / intent に対する別 human 表現を強く罰していない。
- dimensional constraints を削ると gain residual が増え、100% dropout では定義通り `d_feas=1` になる。
- relational constraints を削ると主に leak が増える。`rel_drop100` では leak だけでなく gain も増えており、
  relations を全削除したことで native dimensional response 自体も edit request を十分 drive できなくなるケースがある。
- Vitruvion は dropout controls より median は小さいが、human heldout より明確に大きく、
  その主成分は gain ではなく leak である。これは実験 1 の解釈と整合する。

この control は、`d_feas` が「等価な別表現」をほぼ罰せず、
dimension loss と relation loss を異なる成分で罰することを確認する sanity check である。

なお、ここでは dimensional と relational を独立に drop した。
これは gain term と leak term がそれぞれ期待通り反応するかを切り分けるためである。
両者を同時に drop する mixed control は、自然には `d_feas` が両成分の合成として増えるはずだが、
追加で示すなら appendix の sanity check として十分であり、主張の必須条件ではない。
むしろ mixed dropout だけでは、どちらの成分が効いたかが曖昧になるため、
main table では独立 dropout の方が診断的である。

Artifacts:

- `artifacts/v3_controls_scores.csv`
- `artifacts/v3_controls_details.csv`

### 3. eps / solver tolerance robustness

`d_feas` 本体に入る主要な数値 hyperparameter は、`J_x` と `J_q R_C` の
absolute SVD pseudo-inverse cutoff `eps` である。`eps ∈ {1e-11, 1e-9, 1e-7}` を同一設定で振った。

`eps` の役割をもう少し具体的に書く。`d_feas` ではまず各 sketch の constraint equations
`F(x, p)=0` を線形化し、

```text
J_x dx + J_p dp = 0
R_C = dx / dp = - pinv(J_x) J_p
```

を計算する。さらに、共通 edit probe `q` に対する response は

```text
S_C^(q) = R_C pinv(J_q R_C)
```

で計算する。したがって擬似逆は少なくとも 2 箇所に現れる。

```text
pinv(J_x),  pinv(J_q R_C)
```

実装ではこれらを SVD で計算し、特異値 `sigma` が `sigma >= eps` の成分だけを rank として採用する。
つまり `eps` は、ほぼゼロの特異値に対応する数値的に不安定な方向を
「実際に可動な方向」として拾ってしまわないための absolute rank cutoff である。
`eps` が大きすぎると、本来使うべき弱い response direction を捨ててしまい、
edit attainability を過小評価する可能性がある。逆に `eps` が小さすぎると、
near-singular な方向に巨大な係数を持つ response を許してしまい、
leak や gain が数値ノイズに敏感になる可能性がある。

なお、ここでの `eps` は HistCAD-style finite edit の `delta_scale` や Newton solver tolerance とは別物である。
`eps` は `d_feas` の一次線形 algebra の rank 判定、`delta_scale` は有限編集でどれだけ値を動かすか、
Newton/relation tolerance は有限編集後の成否判定に効く。
このため、`d_feas` の安定性確認では `eps` sweep が本体の robustness check であり、
solver tolerance sweep は source-validation mask が変わっても結論が変わらないかを見る補助確認である。

validated controls:

| eps | heldout med | Vitruvion med | dim50 med | dim100 med | rel50 med | rel100 med | Vit vs heldout |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1e-11 | ~0 | 0.0295 | 0.1667 | 1.0000 | 0.4479 | 0.5529 | h=57, v=8, t=49, p=8.38e-9 |
| 1e-9 | ~0 | 0.0295 | 0.1667 | 1.0000 | 0.4479 | 0.5529 | h=57, v=8, t=49, p=8.38e-9 |
| 1e-7 | ~0 | 0.0295 | 0.1667 | 1.0000 | 0.4479 | 0.5529 | h=57, v=8, t=49, p=8.38e-9 |

ここで `h/v/t` は `d_feas` による Vitruvion vs human heldout の paired 勝敗数であり、
`h` は human heldout が良い rotation 数、`v` は Vitruvion が良い rotation 数である。
この範囲では、median trajectory と paired conclusion は完全に不変だった。

さらに、source validation に使う finite solver tolerance も次で振った。

| validation tolerance | heldout med | Vitruvion med | dim50 med | dim100 med | rel50 med | rel100 med | Vit vs heldout |
|---|---:|---:|---:|---:|---:|---:|---|
| loose: `newton_tol=1e-5`, `relation_tol=1e-2` | ~0 | 0.0295 | 0.1667 | 1.0000 | 0.4479 | 0.5529 | h=57, v=8, t=49, p=8.38e-9 |
| default: `newton_tol=1e-6`, `relation_tol=1e-3` | ~0 | 0.0295 | 0.1667 | 1.0000 | 0.4479 | 0.5529 | h=57, v=8, t=49, p=8.38e-9 |
| strict: `newton_tol=1e-7`, `relation_tol=1e-4` | ~0 | 0.0295 | 0.1667 | 1.0000 | 0.4479 | 0.5529 | h=57, v=8, t=49, p=8.38e-9 |

したがって、今回の結論は少なくとも `eps` と source-validation tolerance の自然な範囲では安定である。
ただし、`d_feas` 本体は有限編集量 `delta_scale` を使わない。
`delta_scale` は HistCAD-style finite edit / source validation 側の hyperparameter であり、
局所線形化の妥当性は次の finite edit check で見る。

Artifacts:

- `artifacts/v3_controls_scores_tol_loose.csv`
- `artifacts/v3_controls_scores_tol_strict.csv`

### 4. finite edit との局所一致

`exp_linear_vs_finite.py` で、native dimensional edit に対する線形応答
`dx_lin = S_C^(q) delta` と、同じ finite edit を Newton solve した後の
`dx_fin` を比較した。これは `d_feas` の直接評価ではなく、first-order response が
finite edit solver と局所的に整合するかを見る診断である。

結果は混合的だった。

raw primitive-space relative error `||dx_fin - dx_lin|| / ||dx_fin||`:

| file / annotator | 0.1% bbox | 0.5% bbox | 1.0% bbox | interpretation |
|---|---:|---:|---:|---|
| `26648...` / Eko | 0.0001 | 0.0435 | 0.0294 | small finite edits agree well |
| `26648...` / Sagar | 0.0002 | 0.0003 | 0.0028 | small finite edits agree very well |
| `122579...` / Sagar | 0.0302 | 0.0336 | 0.0735 | small finite edits mostly agree |
| `122579...` / Ryotaro | 0.9984 | 1.0000 | 1.0197 | finite solve branch / parameterization mismatch |

したがって、linear response は安定な sketch/probe では小さい finite edit とよく一致する。
一方で、全 probe で一様に一致するわけではなく、FreeCAD/Newton roundtrip、branch change、
near-singular parameterization の影響を受ける case がある。これは `d_feas` の主張を壊すというより、
finite edit との橋渡し evidence としては限定的に扱うべき、という示唆である。

論文上は、4 は main evidence ではなく appendix diagnostic に留めるのが安全である。
主張の中心は、実験 2/3 の control と robustness、および実験 1/2 の paired comparison /
dropout trajectory に置くのがよい。

Artifacts:

- `artifacts/exp_linear_vs_finite_26648_v3.png`
- `artifacts/exp_linear_vs_finite_v3.png`

### 5. OES preservation set から Point-Point Coincident を除外する ablation

HistCAD-style OES の cPCSR は preservation set の作り方に依存する。
特に Point-Point Coincident は topology preservation としてほぼ常に満たされやすく、
cPCSR を押し上げる可能性がある。そこで、datalib `Coincident` のうち
Point-Point Coincident だけを preservation set から除外する ablation を行った。
Point-Line / Line-Line 的な `Coincident` は geometric relation として残した。

結果:

| setting | paired n | human-better | Vit-better | tie | z | p |
|---|---:|---:|---:|---:|---:|---:|
| original OES | 114 | 53 | 30 | 31 | 2.74 | 6.1e-3 |
| no Point-Point Coincident | 114 | 54 | 28 | 32 | 3.21 | 1.3e-3 |

validated dropout trajectory:

| setting | drop0 | drop25 | drop50 | drop75 | drop100 |
|---|---:|---:|---:|---:|---:|
| original OES | 0.2857 | 0.5000 | 0.6944 | 0.8750 | 1.0000 |
| no Point-Point Coincident | 0.2500 | 0.5000 | 0.6667 | 0.8571 | 1.0000 |

detail-level では、Point-Point Coincident は多くの rows に含まれていた。

```text
rows joined: 3122
rows with removed Point-Point Coincident: 2918
median removed per row: 6
max removed per row: 17
rows where OES changed: 1267
OES decreased after removal: 1080 rows
OES increased after removal: 187 rows
median(original - no_pp): 0.0362
```

解釈:

- Point-Point Coincident は確かに cPCSR/OES を少し押し上げていた。
- しかし除外しても、HistCAD OES が dimensional dropout を高く評価する failure mode は消えない。
- Vitruvion vs human heldout の paired comparison では、むしろ OES の human 優位が少し強くなる。
- `0/56013_752d02cb_0001_sketch00.FCStd` の定性例は、Point-Point Coincident 除外前後で変化しなかった。
  この case の high OES は Point-Point Coincident inflate ではなく、
  finite edit が通ることと、残った non-Coincident relations が preservation を満たすことに由来する。

したがって、Point-Point Coincident は OES/cPCSR の小さな confound ではあるが、
本稿で示している HistCAD-style OES の failure mode の主因ではない。

Artifacts:

- `artifacts/source_conditioned_dropout_scores_no_ppcoincident.csv`
- `artifacts/source_conditioned_dropout_details_no_ppcoincident.csv`

## Appendix: 実行コマンド

`d_feas` full paired:

```bash
conda run -n freecad python notebooks/exp_constrained_response_metric.py \
  --drop-rates 0,100 \
  --output-csv artifacts/constrained_response_scores_full.csv \
  --details-csv artifacts/constrained_response_details_full.csv
```

Dimensional dropout sweep:

```bash
conda run -n freecad python notebooks/exp_constrained_response_metric.py \
  --drop-rates 0,25,50,75,100 \
  --output-csv artifacts/constrained_response_scores_dropout.csv \
  --details-csv artifacts/constrained_response_details_dropout.csv
```

Evidence aggregation:

```bash
python notebooks/analyze_dalign_evidence.py
python notebooks/analyze_dfeas_trajectory.py
python notebooks/analyze_dfeas_outliers.py
```

Additional controls / robustness:

```bash
conda run -n freecad python notebooks/exp_v3_controls_and_robustness.py \
  --output-csv artifacts/v3_controls_scores.csv \
  --details-csv artifacts/v3_controls_details.csv

conda run -n freecad python notebooks/exp_v3_controls_and_robustness.py \
  --eps-values 1e-9 \
  --newton-tol 1e-5 \
  --relation-tol 1e-2 \
  --output-csv artifacts/v3_controls_scores_tol_loose.csv \
  --details-csv artifacts/v3_controls_details_tol_loose.csv

conda run -n freecad python notebooks/exp_v3_controls_and_robustness.py \
  --eps-values 1e-9 \
  --newton-tol 1e-7 \
  --relation-tol 1e-4 \
  --output-csv artifacts/v3_controls_scores_tol_strict.csv \
  --details-csv artifacts/v3_controls_details_tol_strict.csv
```

Finite edit local check:

```bash
conda run -n freecad python notebooks/exp_linear_vs_finite.py \
  --filename 26648_59fbce6a_0000_sketch03.FCStd \
  --annotators Eko,Sagar \
  --n-probes 5 \
  --out-png artifacts/exp_linear_vs_finite_26648_v3.png

conda run -n freecad python notebooks/exp_linear_vs_finite.py \
  --filename 122579_2663ded3_0000_sketch04.FCStd \
  --annotators Sagar,Ryotaro \
  --n-probes 5 \
  --out-png artifacts/exp_linear_vs_finite_v3.png
```

Point-Point Coincident preservation ablation for OES:

```bash
conda run -n freecad python notebooks/exp_source_conditioned_dropout.py \
  --drop-rates 100,75,50,25,0 \
  --exclude-point-point-coincident-preservation \
  --output-csv artifacts/source_conditioned_dropout_scores_no_ppcoincident.csv \
  --details-csv artifacts/source_conditioned_dropout_details_no_ppcoincident.csv
```

Naive `S_C^(q)` match (`d_smatch`) vs `d_feas`:

```bash
conda run -n freecad python notebooks/exp_smatch_metric.py \
  --drop-rates 0,25,50,75,100 \
  --output-csv artifacts/smatch_vs_dfeas_summary.csv \
  --details-csv artifacts/smatch_vs_dfeas_details.csv
```
