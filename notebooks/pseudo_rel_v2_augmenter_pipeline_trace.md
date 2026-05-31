# pseudo-GT v2 (規則付与器) — npy生成・学習推論パイプラインのトレースと安全性判断

2026-05-31。問い: 規則付与器(`src/sketch_handlers/rule_augmenter.py`)で pseudo-GT を作るのに
`scripts/generate_pseudo_rel_dataset.py` を**素朴に**使ってよいか。relational constraints は
大量に冗長で、その冗長性を持つ GT がパイプラインのどこかで想定外の挙動を起こさないか。

関連: 研究プラン §14.6 / §15、メモリ `rule-augmenter-and-gate`。

---

## 0. 結論(TL;DR)

- **素朴な流用は不可**。理由は2つ:
  1. `generate_pseudo_rel_dataset.py` は *rel-only RLOO モデル* を走らせて関係を予測する
     スクリプトで、規則付与器を一切呼ばない。そのままでは augmenter pseudo-GT を作れない。
  2. 仮に差し替えても、既定の solver 検証 (`:168 if status.Status != 'Converged'`) が
     **Redundant を却下**する。`get_solver_status` は Converged+redundant を `'Redundant'` に
     再分類する(`solver_feedback.py:299–305`)ため、規則付与器の **full closure はほぼ全件が
     Redundant 判定 → ほぼ全却下**になる(コメント「Redundant tolerated」は誤り。メモリ既知)。
- **一方、冗長 GT 自体はダウンストリームで安全**。これは静的トレース＋実データで確認した(§2–§3)。
  - 報酬は GT を **`datalib.Sketch` オブジェクト**(`batch['sketch']`)として受け取り、関係を
    **SVD ランクで motion subspace に縮約**する。冗長行は特異値≈0で**ランクに寄与せず吸収**される。
  - tokenizer の `max_token_length` 切り詰めは **GT subspace 経路に無関係**(後述の二経路)。
- よって判断は **「小変更で可」**。専用スクリプト `scripts/generate_pseudo_rel_augmented_dataset.py`
  を新規作成した(§4)。safety gate 付きで、生成 npy は学習時にクラッシュしないことを保証する。
- 動作確認の **canary (2000件→1990 valid, 0 error)** を `/tmp/sg_pseudo_rel_v2_canary.npy` に生成済み。
  `ConstraintDataset.__getitem__` 経由 800件ロードで **エラー 0** = 学習可能。rare relation が
  GT に豊富に入った(Mirror/Perp/Parallel/Equal/Tangent…)。本生成のスコープのみ要確認(§6)。

---

## 1. 規則付与器とは(この文書だけで読めるよう再掲)

座標から「点Pは線L上か」「線Aと線Bは平行か」等を総当たり検出し、handler.residual≈0 の候補だけを
制約として書き戻す**モデル不要**のプログラム。`augment_sketch(sketch)` が検出制約を加えた
`datalib.Sketch` を返す。**full relational closure**(例: Horizontal2本 と それが含意する Parallel を
両方)を吐くため、出力は意図的に冗長。運用点 scale=0.25(§15.4 較正)。

---

## 2. 静的トレース: 冗長 GT が通る2つの経路

ConstraintDataset.`__getitem__`(`constraint_dataset.py:68`)は1つの sketch から**2つ**を作る:

| 経路 | 中身 | 切り詰め | 冗長の扱い | 消費先 |
|---|---|---|---|---|
| **① GT sketch object** | `result["sketch"]`=full `datalib.Sketch` | **なし**(オブジェクトそのまま) | SVD で吸収 | **報酬**(`ResponseIntentReward` 等) |
| ② tokenized `c_val` | `tokenizer.tokenize()` 出力 | **あり**(`pad_or_truncate(…,192)`) | 末尾を黙って切る | CE/SFT・`gt_tokens` |

### 2.1 報酬は経路①(full object)を使う — 冗長は SVD で吸収

- RLOO: `_compute_rloo_loss` → `build_context(sketch_original_list=expanded['sketch'], …)`
  (`constraint_ar_module_rloo.py:185`)。報酬は `ctx.sketch_original_list[i]` を `gt_sketch` として使う。
- `get_or_compute_gt_motion_relational_subspace` → `motion_relational_subspace(gt_sketch, …)`
  は `gt_sketch.constraints.values()` の**全関係**を `stack_relation_jacobians_at` で J_R に積む
  (`geometry.py:146–156`)。`N_C = null(J_R)`、`W_C = range(J_phi @ N_C)`、basis は
  `orthonormal_range`(`geometry.py:36–40`)で **特異値 ≥ eps の本数だけ**を採る。
  → **冗長行(H+H が含意する Parallel 等)は特異値≈0でランク非寄与 = 吸収**。`dim_gt` は count では
  なく rank なので、冗長を何本足しても不変。ユーザの仮説どおり。
- **重要**: GT subspace は経路①(full object)なので、tokenizer の 192 切り詰めの影響を**受けない**。
  どれだけ長い closure でも全関係が J_R に入る。

### 2.2 切り詰め(経路②)が効く先と、現 config での無害性

- **CE/SFT**: `total_loss = rl_loss + ce_loss * sft_coeff`(`constraint_ar_module_rloo.py:357`)。
  `sft_coeff` 既定 **0.0**(`configs/model/constraint_ar_rloo.yaml:62`、scaffold config で未上書き)
  → CE は学習に**寄与せず**、切り詰めは無害。
- **DimParsimonyReward**(`gt_tokens` 消費): GT の**寸法**トークン数を数える。pseudo-GT は
  relational-only なので寸法=0 → 切り詰めと無関係。
- **GTMatchReward**: scaffold config の rewards に**不在**(SolverReward + DimParsimony +
  RelationalClassPrior + ResponseIntent のみ)。
- 結論: **現 response-metric RLOO 設定では 192 切り詰めは報酬・学習に実害なし**。
  （SFT/pretrain で v2 を教師に使う、または `gt_tokens` 系報酬を足す場合のみ要対応 → §5 caveat。）

### 2.3 cleanse は制約**間**の冗長を消さない(=意図通り保存)

`cleanse_sketch_constraints` の dedup は**単一制約内のref重複**のみ(`sketch_utils.py:721–744`、
例 Coincident(A,A))。制約**間**の幾何冗長(H+H+Parallel)は消さない。よって冗長は GT に残り、
報酬側 SVD で吸収される、という設計が成立。

### 2.4 正規化は相似変換 = 検出述語は不変

`normalize_sketch` = center + **等方** rescale(`sketch_utils.py:248–253`、単一 scale_factor)。
Horizontal/Vertical/Parallel/Perp/Coincident/Concentric/Equal/Tangent/Mirror は相似変換不変 →
**augment を正規化の前後どちらで行っても同じ集合**。geometry 不一致の懸念なし。

---

## 3. 実データ検証(`/tmp/probe_augmenter_pipeline.py`、freecad env、60件)

`sg_filtered_unique_valid.npy` 先頭60件で、base(関係のみ)→ augment(scale0.25)→
to/from_sequence 往復 → dataset 前処理 を実行:

| 検証項目 | 結果 | 意味 |
|---|---|---|
| **往復後 dim == augment 後 dim** | **59/59 一致** | npy 詰め込み＋ロードで GT subspace が**厳密保存** |
| **augment 後 dim ≤ base dim** | **常に True** | 冗長を平均61本(最大367本)足しても dim は blow up せず縮むだけ |
| augment 後の dim 平均 | base 8.54 → aug 5.49 | 関係が増え motion が締まる(rank 有界) |
| 往復で関係数が減る量 | 平均 −6.2(最大 −17) | cleanse が**冗長**関係を一部削除。**ただし dim は不変**=削られたのは inert な冗長 |
| **エラー** | 1/60 | sketch54。augment 前の**元データ** c_41(3-ref Midpoint 厳密性)。augmenter 由来ではない |
| augment 時間 | 平均 75ms / p95 282ms | 後述スコープ算定の根拠 |

→ 「冗長 GT が pipeline を壊す」懸念は**経路①(報酬)について実証的に否定**された。

---

## 4. 必要だった「小変更」: 専用生成スクリプト

`scripts/generate_pseudo_rel_augmented_dataset.py`(新規)。`generate_pseudo_rel_dataset.py` との差分:

1. **モデル不使用**。各 sketch を `sketch_from_sequence → normalize → cleanse_primitives →
   (既定)全制約strip → augment_sketch(scale0.25) → lenient cleanse/complete → sketch_to_sequence`。
   - 既定は **pure relational closure**(bare primitives から復元)。v1 が「モデル予測の関係で
     置換」だったのと整合。`--keep_original` で既存関係に上乗せも可。
2. **solver 検証は既定 OFF**。`--solver_validate` 指定時のみ、**緩和版**(Redundant 許容、
   Conflicting/Malformed/Distorting のみ却下)。§0 の Redundant 誤却下を回避。
3. **safety gate(常時)**: 生成系列を **dataset と同一の STRICT 前処理**
   (`cleanse_sketch_constraints` 厳密＋`complete_constraint_values` 厳密)で再ロードし、
   `motion_relational_subspace` まで計算。**例外を出す標本/関係0の標本は捨てる**。
   → 保存 npy は **学習時クラッシュしないことを構成的に保証**。
4. `--no_mirror`(Mirror 除外=full_no_mirror ablation)、`--shard i/n`(並列)、`--limit`、`--types`。

### 4.1 canary 実行結果(先頭2000件)
```
valid 1990 / no_rel 10 / load_reject 0 / augment_error 0 / solver_reject 0
avg relations added/sketch 67.9   8.8 sketch/s
```
- `load_reject 0` = safety gate を全件通過。pure closure で**元データの不良制約が消える**ため、
  §3 の sketch54 系エラーも再発しない。

### 4.2 ConstraintDataset 経由ロード(`/tmp/verify_v2_dataset.py`、rloo context, 800件)
```
__getitem__ ok=800  errors=0          ← align=4 の >3-ref guard でもクラッシュ無し = 学習可能
relations/sketch (strict cleanse後) mean 71.2 / max 464
c_val が 192 cap 到達(切り詰め): 430/800 (54%)   ← 経路②のみ。報酬には無関係(§2.2)
relation HEAD-type: Mirror 11617, Coincident 11358, Perpendicular 10461,
  Parallel 8798, Equal 6004, Tangent 2936, Horizontal 2556, Vertical 2536, Midpoint 661
```
→ 本来「ほぼ無い」だった **Parallel/Perp/Equal/Tangent/Mirror が GT に豊富**に入った
(動機 §obs.2 を直接充足)。

---

## 5. 注意点・既知の caveat

- **C5-a 切り詰め(経路②)**: full closure は 54% の sketch で c_val が 192 を超える(最大 1188 tok)。
  **現設定では無害**(§2.2)。ただし将来 v2 で **SFT/pretrain**(`sft_coeff>0`)や **GTMatch/gt_tokens
  系報酬**を使うなら、`max_token_length` を上げるか型別に密度を間引く必要がある。
- **C5-b Concentric → Coincident 正規化**: `cleanse_sketch_constraints` の
  `_convert_concentric_to_coincident`(`sketch_utils.py:770`)が **Concentric を中心同士の
  Coincident に変換**する。よって GT に "Concentric" **ラベルは出ない**(canary でも head-type 0)。
  - これは**脱落ではなく相対ラベル変換**で、**幾何・motion subspace は等価**(中心一致=同心)。
    report の検証で raw Concentric 候補は 3000件中850件 検出されており、cleanse 段で Coincident に
    畳まれているだけ。**§14.4 が言う「Concentric の構造的欠落」の正体はこれ**で、v1 も含む既存挙動。
  - 報酬(motion subspace)には無害。ただし **class prior の `Concentric` 係数や型別の質的集計**には
    "Concentric" が現れない点に注意(必要なら cleanse の変換を gate するか、Coincident-of-centers を
    Concentric として再分類する)。
- **C5-c Mirror は survive**(canary 11617件)。§14.4 の Mirror 欠落は v1(モデルが出せない)由来で、
  augmenter 経由なら GT に入りパイプラインを通る。ただし fiverr precision 0.006 なので
  **full_no_mirror ablation を必ず併走**(§15.5)。
- **C5-d 関係過多のリスク**: closure は dim を強く締める(8.5→5.5)。これは「人間 GT の sparse
  annotation を closure に近づける」狙い(§15.2-4)に沿うが、over-closure(`Equal.length` 全ペア、
  遠点 Mirror 等)が motion subspace を**過剰に**狭める可能性は残る。`full_all` / `full_no_mirror` /
  `high_precision_only` を分けて relation Jacobian rank / null-space / d_H_angle(対 human)で診断する
  (§15.4 方針)。本トレースは**安全性**を確認したもので、**有効性**(d_H_angle 改善)は別途要評価。

---

## 6. 本生成: sharded parallel

- **スコープ確定**: 入力 `sg_filtered_unique_valid.npy`(**1,670,450件**)の **先頭 167k**
  (v1 `sg_pseudo_rel_167k` と同スケール)。variant = **full_all** と **full_no_mirror** の2本。
  scale=0.25。全 1.67M(~53h)は当面の実験に過剰なので採らない。
- **オーケストレータ**: `scripts/run_pseudo_rel_v2_gen.sh`。各 variant を **NSHARD=12** に分割し、
  両 variant を**同時**に走らせる(=24 python worker)。`--shard i/12` は **limit 先・shard 後**
  なので 12 本の和集合がちょうど先頭 167k(各 ~13,917件)。完了後、各 variant の shard を
  `--merge_inputs` で1ファイルにマージ(行順保存で sequence↔metadata の整合維持)。
- **環境条件**(実測): 32 コア / 空き 107GB / 入力 npy は `mmap_mode='r'`(各 worker が 2GB を
  RSS で抱えない)。worker は `torch.set_num_threads(1)` + `OMP/MKL/OPENBLAS=1` で 1 スレッド化し、
  24 本を全コアにばらまく。実測 worker affinity=0–31(フル)、各 ~100% CPU。
  → **ETA ~30 分**(単一プロセス 5.3h を短縮)。
- **出力**: `data_processed/sg_pseudo_rel_v2_full_all.npy` / `data_processed/sg_pseudo_rel_v2_no_mirror.npy`。
- **完全デタッチ起動(セッションを閉じても継続)**:
  ```bash
  setsid bash scripts/run_pseudo_rel_v2_gen.sh < /dev/null \
      > /tmp/gen_v2_orchestrator.log 2>&1 & disown
  ```
  `setsid` で新セッションリーダ化 → 起動した CLI セッションのプロセスグループ signal の影響を
  受けず、**会話セッション/ターミナルを閉じても走り続ける**。
- **監視 / 完了確認**:
  ```bash
  tail -n 20 /tmp/gen_v2_orchestrator.status   # マイルストーン(launching / merged / ALL DONE)
  ls /tmp/gen_v2_DONE                           # 存在すれば全完了
  grep -h valid= /tmp/gen_v2_full_all.shard*.log | tail   # shard 進捗
  ```
- 完了後の検証(自動ではない、手動 or 再開時に実行): 両 npy を `ConstraintDataset` で全件ロードし
  `__getitem__` エラー 0 と relation 分布(canary 同様 Mirror/Perp/Parallel/Equal/Tangent が豊富)
  を確認する。

### 6.1 完了・全件 sanity check (2026-05-31)

本生成は完了。`/tmp/gen_v2_DONE` が存在し、status は
`2026-05-31 11:04:00 ALL DONE`。merge 後の件数は両 variant とも **166,254**。
入力先頭167,000件のうち **746件は no_rel** として棄却され、`load_reject=0`,
`augment_error=0`, `solver_reject=0`。

`ConstraintDataset` の RLOO 実経路(`max_token_length=192`, align=4,
`PrimitiveNoise(std=0.15)`)で全件 `__getitem__` を走査:

| variant | `__getitem__` | rel/sketch mean | rel/sketch max | c_val 192 cap |
|---|---:|---:|---:|---:|
| full_all | 166,254 ok / **0 errors** | 69.4 | 2,936 | 87,558 / 166,254 |
| full_no_mirror | 166,254 ok / **0 errors** | 55.2 | 1,232 | 74,234 / 166,254 |

主要 relation head-type (全件):

| type | full_all | full_no_mirror |
|---|---:|---:|
| Mirror | 2,366,079 | 0 |
| Coincident | 2,283,136 | 2,283,136 |
| Perpendicular | 2,088,540 | 2,088,540 |
| Parallel | 1,751,349 | 1,751,349 |
| Equal | 1,244,472 | 1,244,472 |
| Tangent | 638,475 | 638,475 |
| Horizontal / Vertical | 518,630 / 509,642 | 518,630 / 509,642 |
| Midpoint | 137,427 | 137,427 |

metadata 長も `sequences/sketch_ids/sequence_lengths = 166,254` で一致。full_no_mirror は
Mirror が 0 で、それ以外の relation 分布は full_all と一致するため、Mirror ablation として
意図通り使える。結論: **両 npy は training-ready**。

---

## 7. 次手

1. ✅ パイプライン安全性: 確認済み(冗長 GT は報酬経路で安全)。
2. ✅ 小変更スクリプト + safety gate + merge: 実装・smoke 検証済み。
3. ✅ 本生成(full_all + no_mirror @167k): 完了。全件 sanity check 0 errors(§6.1)。
4. scaffold-v2 + intent を **v2 GT** で回し、§14.6 ゲートで判定(Parallel がサンプルに乗るか /
   leak・response_full_angle が v1 比改善か / 残差が報酬バランス #3 か)。**C5-d** の
   rank/null-space/d_H_angle(対 human)over-closure 診断を併走。`full_all` 対 `full_no_mirror` で
   Mirror over-completion の影響を切り分ける。
5. (config) `train_sequence_file` を v2 npy に差し替え。SFT/gt_tokens を使う場合のみ §C5-a の
   `max_token_length` 対応が要る(現 RLOO 設定では不要)。
