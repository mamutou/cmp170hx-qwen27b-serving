#!/usr/bin/env bash
# 27B 编译缓存快照 / 恢复：把"冷缓存 195s"变成"缓存还在 27s"。
#   hq_cache_snapshot.sh save      # 停止 unit 后执行（引擎在写时会抓到半个文件）
#   hq_cache_snapshot.sh restore   # 缓存丢失/被清后恢复
#   hq_cache_snapshot.sh status
# 用 rsync 而不是 tar：tar 打包大目录曾出现静默丢文件（6567→5056），rsync 有 -n 校验。
set -euo pipefail

SNAP_DIR="${SNAP_DIR:-$HOME/cache-snapshots}"
DST="$SNAP_DIR/vllm.cache"
STAMP_FILE="$SNAP_DIR/vllm.cache.stamp"
VP="$HOME/venv-hq/lib/python3.12/site-packages/vllm"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-27B-W8A16}"
SRC_DIRS=("$HOME/.cache/vllm" "$HOME/.triton" "$HOME/.cache/flashinfer" "$HOME/.cache/torch")

stamp() {
  local ver launch patched cfg
  ver=$(grep -oE '__version__ = "[^"]+"' "$VP/version.py" 2>/dev/null | cut -d'"' -f2 || true)
  launch=$(sha256sum "$HOME/hq_prod.sh" 2>/dev/null | cut -c1-16 || true)
  patched=$(sha256sum "$VP/v1/attention/backends/flash_attn.py" 2>/dev/null | cut -c1-16 || true)
  cfg=$(sha256sum "${MODEL_DIR:-/models/Qwen3.8-27B-W8A16}/config.json" 2>/dev/null | cut -c1-16 || true)
  echo "vllm=${ver:-?} launch=${launch:-?} patched=${patched:-?} cfg=${cfg:-?}"
}

count() { find "$1" -type f 2>/dev/null | wc -l; }
bytes() { du -sb "$1" 2>/dev/null | cut -f1; }

busy() { systemctl --user is-active --quiet vllm-27b-hq.service 2>/dev/null && echo yes || echo no; }

case "${1:-status}" in
  save)
    [ "$(busy)" = "yes" ] && { echo "拒绝：unit 还在跑，会抓到写了一半的缓存。先 systemctl --user stop vllm-27b-hq"; exit 1; }
    mkdir -p "$DST"
    for d in "${SRC_DIRS[@]}"; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      rsync -a --delete "$d/" "$DST/$name/"
    done
    # 校验：文件数与字节数必须一致
    for d in "${SRC_DIRS[@]}"; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      a=$(count "$d"); b=$(count "$DST/$name")
      ab=$(bytes "$d"); bb=$(bytes "$DST/$name")
      if [ "$a" != "$b" ]; then
        echo "校验失败 $name: 源 $a 文件 / 副本 $b 文件"; exit 3
      fi
      printf "  %-12s %6s 文件  %8.1f MB  %s\n" "$name" "$a" \
        "$(echo "scale=1; $bb/1048576" | bc)" \
        "$([ "$ab" = "$bb" ] && echo 字节一致 || echo "字节差 $((ab-bb))")"
    done
    stamp > "$STAMP_FILE"
    echo "快照完成: $DST ($(du -sh "$DST" | cut -f1))  $(cat "$STAMP_FILE")"
    ;;
  restore)
    [ -d "$DST" ] || { echo "没有快照: $DST"; exit 1; }
    [ "$(busy)" = "yes" ] && { echo "警告：unit 在跑，先 stop"; exit 1; }
    want=$(cat "$STAMP_FILE" 2>/dev/null || true); now=$(stamp)
    [ "$want" != "$now" ] && { echo "警告：快照与当前环境不一致"; echo "  快照: $want"; echo "  当前: $now"; }
    for d in "${SRC_DIRS[@]}"; do
      name=$(basename "$d")
      [ -d "$DST/$name" ] || continue
      mkdir -p "$d"
      rsync -a --delete "$DST/$name/" "$d/"
      echo "  恢复 $name: $(count "$d") 文件"
    done
    echo "恢复完成"
    ;;
  status)
    echo "快照: $([ -d "$DST" ] && du -sh "$DST" | cut -f1 || echo 无)  戳记: $(cat "$STAMP_FILE" 2>/dev/null || echo 无)"
    echo "当前: $(stamp)   unit 在跑: $(busy)"
    for d in "${SRC_DIRS[@]}"; do [ -d "$d" ] && printf "  %-30s %6s 文件  %s\n" "$d" "$(count "$d")" "$(du -sh "$d" | cut -f1)"; done
    ;;
  *) echo "用法: $0 {save|restore|status}"; exit 2 ;;
esac
