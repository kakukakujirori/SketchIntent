#!/usr/bin/env python3
"""
夜間自動実行スクリプト
- GPU空き確認（4台すべて）
- 処理済みデータのスキップ
- 朝11時に自動停止

使い方:
  python night_runner.py

Cron設定例 (毎日2時に実行):
  crontab -e で以下を追加:
  0 2 * * * cd /home/ryotaro/my_works/My-Vitruvion/notebooks && /usr/bin/python3 night_runner.py >> night_runner.log 2>&1
"""

import subprocess
import os
import glob
import time as time_module
from datetime import datetime, time
import signal
import sys

OUTPUT_DIR = "qwen3_refined_data_with_solver"
SCRIPT_PATH = "v20260124_qwen3_refine_data_with_solver.py"
REQUIRED_GPUS = 4
MEMORY_THRESHOLD_MB = 10000  # このMB以下なら未使用とみなす（Xorgのみ≒5MB）
START_HOUR = 1   # 開始時刻 (1時) - cronがこの時刻に起動
END_HOUR = 11    # 終了時刻 (11時) - この時刻を過ぎたら停止
CHECK_INTERVAL_SEC = 60  # 時間チェック間隔
CONDA_ENV = "freecad"  # アクティベートするconda/mamba環境
# ==================================


def log(msg: str):
    """タイムスタンプ付きログ出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def check_gpu_availability() -> bool:
    """全GPUが空いているかチェック（メモリ使用量で判定）"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            log(f"nvidia-smi failed: {result.stderr}")
            return False

        memory_usage = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        log(f"GPU memory usage (MB): {memory_usage}")

        if len(memory_usage) < REQUIRED_GPUS:
            log(f"Only {len(memory_usage)} GPUs found, need {REQUIRED_GPUS}")
            return False

        # 全GPUのメモリ使用量がしきい値以下なら空いていると判断
        available = all(usage < MEMORY_THRESHOLD_MB for usage in memory_usage[:REQUIRED_GPUS])
        if not available:
            busy_gpus = [i for i, usage in enumerate(memory_usage[:REQUIRED_GPUS]) if usage >= MEMORY_THRESHOLD_MB]
            log(f"GPUs {busy_gpus} are busy")

        return available

    except subprocess.TimeoutExpired:
        log("nvidia-smi timed out")
        return False
    except Exception as e:
        log(f"GPU check failed: {e}")
        return False


def get_next_index() -> int:
    """処理済みインデックスの次を返す（両方のファイルが存在する最大インデックス + 1）"""
    commands_dir = os.path.join(OUTPUT_DIR, "commands")
    if not os.path.exists(commands_dir):
        return 0

    pattern = os.path.join(commands_dir, "*_original.py")
    files = glob.glob(pattern)

    if not files:
        return 0

    # 両方のファイル（original + refined）が存在するインデックスを収集
    completed_indices = []
    for f in files:
        basename = os.path.basename(f)
        try:
            idx = int(basename.split('_')[0])
            refined = f.replace('_original.py', '_refined.py')
            if os.path.exists(refined):
                completed_indices.append(idx)
        except ValueError:
            continue

    if not completed_indices:
        return 0

    return max(completed_indices) + 1


def is_within_time_window() -> bool:
    """現在時刻が実行可能な時間帯内かチェック

    例: START_HOUR=1, END_HOUR=11 の場合
    - 1:00 〜 10:59: True (実行可能)
    - 11:00 〜 0:59: False (実行不可)
    """
    now = datetime.now().time()
    start = time(START_HOUR, 0)
    end = time(END_HOUR, 0)

    if start <= end:
        # 通常の範囲 (例: 1:00〜11:00)
        return start <= now < end
    else:
        # 深夜をまたぐ範囲 (例: 22:00〜6:00)
        return now >= start or now < end


def run_training(start_idx: int) -> subprocess.Popen:
    """メインスクリプトを環境アクティベート経由で起動"""
    cuda_devices = ','.join(str(i) for i in range(REQUIRED_GPUS))

    # mamba run を使用して環境内で直接実行
    # mamba run は activate なしで環境内コマンドを実行できる
    cmd = [
        '/mnt/hdd1/ryotaro/miniforge3/bin/mamba', 'run', '-n', CONDA_ENV,
        'python', SCRIPT_PATH,
        '--start_idx', str(start_idx),
        '--output_dir', OUTPUT_DIR
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_devices

    log(f"Starting with CUDA_VISIBLE_DEVICES={cuda_devices}")
    log(f"Environment: {CONDA_ENV}")
    log(f"Command: {' '.join(cmd)}")

    return subprocess.Popen(
        cmd,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )


def main():
    log("=" * 60)
    log("Night Runner Started")
    log(f"Will stop at {END_HOUR}:00")
    log("=" * 60)

    # GPU空き確認
    if not check_gpu_availability():
        log("GPUs are not all available. Exiting.")
        return

    # 開始インデックスを取得
    start_idx = get_next_index()
    log(f"Resuming from index {start_idx}")

    # メインスクリプトを実行
    process = run_training(start_idx)

    # シグナルハンドラ設定（Ctrl+C等で子プロセスも終了）
    def signal_handler(signum, frame):
        log(f"Received signal {signum}. Terminating child process...")
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 監視ループ
    while process.poll() is None:
        # 時間帯チェック
        if not is_within_time_window():
            log(f"Outside time window ({START_HOUR}:00 - {END_HOUR}:00). Stopping...")
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                log("Force killing process...")
                process.kill()
            break

        time_module.sleep(CHECK_INTERVAL_SEC)

    exit_code = process.returncode
    log(f"Process finished with exit code: {exit_code}")
    log("Night Runner Finished")


if __name__ == "__main__":
    main()
