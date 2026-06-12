# eda_code

論文の実験セクションの数値を得るためのコードです。
データ入力は repo root の `data_eval/` に固定されています。

`data_eval/<batch>/` には次が同梱されている前提です。

- `<sketch_id>.py`: annotation 前の入力 sketch
- `<sketch_id>_<label>.FCStd`: annotation 済み sketch

## 責務

| module | 責務 |
|---|---|
| `config.py` | `data_eval/` 固定パス、annotation label、共通定数。 |
| `manifest.py` | `data_eval/` の走査、除外規則、complete trio 判定。 |
| `canonical.py` | `data_eval/<batch>/<sketch_id>.py` の lookup、sketch load、canonical geometry remap。 |
| `file_bundle.py` | annotation `.FCStd` と model `.py` を canonical geometry に remap した `FileBundle` を作る。 |
| `bundle_loader.py` | `data_eval/` manifest 形式から `FileBundle` 構築引数へ変換する薄い adapter。 |
| `dataset_features.py` | `.FCStd` から dataset statistics 用 feature を抽出。 |
| `controls.py` | synthetic degradation control、現在は relational dropout。 |
| `motion_metrics.py` | 論文の `theta_full`, `theta_rel`, `asgn_sim` 実装。 |
| `scripts/` | 上から順に実行する CLI entrypoints。 |

## データと環境

すべて repo root から `freecad` env で実行してください。

データルートを切り替える CLI option はありません。常に `./data_eval` を使います。

モデル推論結果を評価する場合は、`config.py` の `MODEL_OUTPUT_ROOTS` を編集してください。各 root は `data_eval/` と同じく `0` から `11` の subdirectory を持ち、各 batch directory の中に `<sketch_id>.py` が入っている必要があります。

```python
MODEL_OUTPUT_ROOTS = {
    "base": REPO_ROOT / "model_outputs" / "base",
    "solver_rl": REPO_ROOT / "model_outputs" / "solver_rl",
}
```

設定した root が存在しない場合は warning を出して model 行なしで続行します。root が存在するのに評価対象 batch/file の `.py` が欠けている場合は、欠損した推論結果として error で停止します。

出力先は `fusion360_annotation/eda_code/out/` に統一しています。中間 JSON、図、CSV shard、集計入力を同じ場所に置くことで、後段スクリプトの指定を単純にしています。

前回の実行結果をクリーンにするには、生成物がまとまっている `fusion360_annotation/eda_code/out/` を削除してください。各スクリプトは必要な出力ディレクトリを再作成します。

```bash
rm -rf fusion360_annotation/eda_code/out
```

長時間走るスクリプトは `tqdm` が入っていれば progress bar を表示します。`tqdm` がない環境では、一定間隔の進捗ログに fallback します。

## 上から実行する順番

### 0. manifest sanity check

`data_eval/` を走査し、complete trio 数、除外数、canonical `.py` 欠落を確認します。

```bash
python fusion360_annotation/eda_code/scripts/00_check_manifest.py
```

### 1. dataset statistics extraction

全 annotation `.FCStd` を読み、section 3 / appendix 用の per-annotation statistics を抽出します。

```bash
python fusion360_annotation/eda_code/scripts/01_extract_dataset_stats.py \
  --out fusion360_annotation/eda_code/out
```

出力:

- `out/records.json`
- `out/summary.json`

### 2. section 3 figure

`records.json` から section 3 の constraint-type distribution 図と `ch3_stats.json` を作ります。

```bash
python fusion360_annotation/eda_code/scripts/02_plot_ch3_distributions.py \
  --out fusion360_annotation/eda_code/out
```

出力:

- `out/ch3_constraint_distribution.pdf`
- `out/ch3_constraint_distribution.png`
- `out/ch3_stats.json`

### 3. appendix dataset statistics

`records.json` から appendix 用の dataset statistics と補助図を作ります。

```bash
python fusion360_annotation/eda_code/scripts/03_appendix_stats.py \
  --out fusion360_annotation/eda_code/out
```

出力:

- `out/appendix_stats.json`
- `out/appendix_per_annotator.png`
- `out/appendix_count_hist.png`

### 4. section 5 token-vs-motion extraction

human-human pair ごとの token-F1 と motion disagreement を `ch5_pairs.json` に抽出します。

```bash
python fusion360_annotation/eda_code/scripts/04_extract_ch5_pairs.py \
  --out fusion360_annotation/eda_code/out
```

出力:

- `out/ch5_pairs.json`
- `out/ch5_summary.json`

### 5. section 5 token-vs-motion figure

`ch5_pairs.json` から section 5 の token-vs-motion 図と `ch5_stats.json` を作ります。

```bash
python fusion360_annotation/eda_code/scripts/05_plot_ch5.py \
  --out fusion360_annotation/eda_code/out \
  --layout column
```

出力:

- `out/ch5_token_vs_motion_column.pdf`
- `out/ch5_token_vs_motion_column.png`
- `out/ch5_stats.json`

### 6. main motion metrics

leave-one-out rotations と dropout controls を `theta_full` / `theta_rel` / `asgn_sim` で評価します。

```bash
python fusion360_annotation/eda_code/scripts/06_run_motion_eval.py \
  --out fusion360_annotation/eda_code/out/motion_eval.csv
```

出力:

- `out/motion_eval*.csv`
- `out/motion_eval*_model_status.csv`

### 7. HistCAD OES baseline

同じ rotations / variants (`source_self`, `heldout`, dim-drop, rel-drop, model) に対して HistCAD OES baseline を有限編集 solver で評価します。
OES は有限編集 solver を大量に呼ぶので重いです。batch shard 推奨です。

全 batch を一括で実行する場合:

```bash
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --out fusion360_annotation/eda_code/out/oes_eval.csv
```

全 batch を 2 batch ごとの shard で実行する場合:

```bash
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 0 1 \
  --out fusion360_annotation/eda_code/out/oes_eval_b0_1.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 2 3 \
  --out fusion360_annotation/eda_code/out/oes_eval_b2_3.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 4 5 \
  --out fusion360_annotation/eda_code/out/oes_eval_b4_5.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 6 7 \
  --out fusion360_annotation/eda_code/out/oes_eval_b6_7.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 8 9 \
  --out fusion360_annotation/eda_code/out/oes_eval_b8_9.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 10 11 \
  --out fusion360_annotation/eda_code/out/oes_eval_b10_11.csv
```

並列実行したい場合は各コマンドの末尾に `&` を付けて最後に `wait` してください。ただし OES はかなり重いので、同時実行数はマシンに合わせて絞ってください。

複数 shard は Step 11 の `11_analyze_paper_numbers.py` が `out/oes_eval*.csv` を glob してまとめて読みます。個別の merge script は不要です。

出力:

- `out/oes_eval*.csv`

### 8. token-F1 baseline

同じ rotations / variants に対して constraint token-F1 baseline を評価します。

```bash
python fusion360_annotation/eda_code/scripts/08_run_token_f1.py \
  --out fusion360_annotation/eda_code/out/token_f1.csv
```

出力:

- `out/token_f1.csv`

### 9. human agreement

human-human の dimensional placement Jaccard と token-F1 を計算します。

```bash
python fusion360_annotation/eda_code/scripts/09_run_agreement.py \
  --out fusion360_annotation/eda_code/out/agreement.csv
```

出力:

- `out/agreement.csv`

### 10. toy placement demo

`S_place` が dimensional placement 軸を拾うことを square / rectangle toy で確認します。

```bash
python fusion360_annotation/eda_code/scripts/10_run_toy_placement.py
```

### 11. aggregate paper numbers

`out/` の CSV shards を集計し、論文表・appendix 用の数値を表示します。`motion_eval*.csv`、`oes_eval*.csv`、`token_f1*.csv`、`agreement.csv` はこのスクリプトが直接読みます。最初に `paper/sec/5_experiments.tex` の Table 形式に合わせて、Ours (`theta_full` / `theta_rel` / `S_place`) と Baselines (OES / F1) を横並びで表示します。

```bash
python fusion360_annotation/eda_code/scripts/11_analyze_paper_numbers.py \
  --out-dir fusion360_annotation/eda_code/out
```

## 注意

- この版は「metric を変える」ためのものではなく、「完成済み `data_eval/` を唯一の dataset contract にする」ためのものです。
- 既存論文図を更新する場合は、生成した PDF を明示的に `paper/figs/` へコピーしてください。
