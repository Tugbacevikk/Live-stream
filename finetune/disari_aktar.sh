#!/usr/bin/env bash
# Eğitilen modeli sherpa-onnx biçimine (int8 ONNX) çevirip sistemin okuyacağı klasöre koyar.
#   bash finetune/disari_aktar.sh ~/ft ft_exp 10 3 /mnt/c/.../models/vi-ince-ayar-2026-11
#   (kök, exp-dir, son epoch, ortalama alınacak epoch sayısı, hedef klasör)
set -euo pipefail
ROOT=$1; EXP=$2; EPOCH=$3; AVG=$4; OUT=$5
source "$ROOT/venv/bin/activate"
export PYTHONPATH=$ROOT/icefall:${PYTHONPATH:-}
BPE=$ROOT/temel_model/data/Vietnam_bpe_2000_new
RECIPE=$ROOT/icefall/egs/librispeech/ASR/zipformer

# finetune.py epoch-N.pt kaydeder; export-onnx son AVG epoch'un ortalamasını alır
python "$RECIPE/export-onnx.py" --exp-dir "$EXP" --tokens "$BPE/tokens.txt" \
    --epoch "$EPOCH" --avg "$AVG" --use-averaged-model 1 --causal False

mkdir -p "$OUT"
cp "$EXP"/encoder-epoch-"$EPOCH"-avg-"$AVG".int8.onnx "$EXP"/joiner-epoch-"$EPOCH"-avg-"$AVG".int8.onnx \
   "$EXP"/decoder-epoch-"$EPOCH"-avg-"$AVG".onnx "$BPE/tokens.txt" "$BPE/bpe.model" "$OUT/"
# hotword için bpe.vocab (yoksa sherpa-onnx hotword'leri sessizce yok sayar)
python - "$OUT" <<'PY'
import sys, sentencepiece as spm
out = sys.argv[1]
sp = spm.SentencePieceProcessor(model_file=f"{out}/bpe.model")
with open(f"{out}/bpe.vocab", "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
PY
cat > "$OUT/MODEL_KARTI.json" <<JSON
{"temel": "zzasdf/viet_iter3_pseudo_label exp/epoch-12-avg-8.pt (Zipformer-68M, Apache-2.0)",
 "ince_ayar_dizini": "$EXP", "epoch": $EPOCH, "avg": $AVG,
 "tarih": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "icefall": "$(git -C "$ROOT/icefall" rev-parse --short HEAD)"}
JSON
echo "Apache-2.0 (temel model); eğitim verisinin lisansı ayrıca geçerli" > "$OUT/LICENSE"
echo "Tamam: $OUT  ->  Kapı 2: python finetune/kapi2.py --yeni $OUT ..."
