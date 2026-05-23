#!/bin/bash
# デモ録画: asciinema rec demo.cast → svg に変換
# 使い方: bash record_demo.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 出力をゆっくり流す (各行を 30ms 間隔で表示)
slow_cat() {
    while IFS= read -r line; do
        printf '%s\n' "$line"
        sleep 0.04
    done
}

# キャプチャ: cast ファイル生成
~/.local/bin/asciinema rec demo.cast \
    --overwrite \
    --command "bash -c 'python3 collapse_radar.py --no-llm --scenario S1,S5 --quiet | slow_cat; sleep 2'"

echo "→ demo.cast 生成完了"
echo "  asciinema upload demo.cast  でオンライン共有"
echo "  または demo.cast を GitHub に commit して README に埋め込む"
