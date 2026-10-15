#!/usr/bin/env bash
# İnce ayar ortamı (WSL2 Ubuntu 22.04 + NVIDIA GPU). Bir kez çalıştırılır.
#   bash finetune/kurulum.sh ~/ft          # her şey ~/ft altına
# Gerekenler: Windows'ta güncel NVIDIA sürücüsü (WSL2 CUDA desteği), WSL2'de `nvidia-smi` çalışmalı.
set -euo pipefail
ROOT=${1:-$HOME/ft}
ICEFALL_COMMIT=c1eda5427c46502b2cfdcbeb910bf35f2134a0d4   # 8 Ekim 2026; denenmiş sürüm
mkdir -p "$ROOT" && cd "$ROOT"

# Betik Ubuntu 22.04 için (Python 3.10: k2'nin hazır paketleri var). Başka dağıtımda
# (ör. Kali: daha yeni Python) k2 paketi bulunamayabilir. Windows'ta: wsl --install -d Ubuntu-22.04
if ! grep -q 'VERSION_ID="22.04"' /etc/os-release 2>/dev/null; then
  echo "UYARI: Ubuntu 22.04 değil ($(. /etc/os-release; echo "$PRETTY_NAME")). Önerilen: wsl --install -d Ubuntu-22.04"
  read -r -p "Yine de devam edilsin mi? [e/H] " ans; [ "$ans" = "e" ] || exit 1
fi
nvidia-smi > /dev/null || { echo "GPU görünmüyor: Windows NVIDIA sürücüsünü güncelleyin"; exit 1; }

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -U pip
# torch + k2 sürümleri birbirine bağlı: k2 sayfasındaki eşleşmeye göre (CUDA 12.1, torch 2.4.0)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# k2: sayfadaki tekerleklerden torch/CUDA/Python sürümüne uyan en yenisi seçilir
PYTAG=cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
K2_URL=$(curl -s https://k2-fsa.github.io/k2/cuda.html | grep -o 'https://[^"]*cuda12\.1\.torch2\.4\.0-'"$PYTAG"'-'"$PYTAG"'-[^"]*linux_x86_64\.whl' | sort -V | tail -1)
[ -n "$K2_URL" ] || { echo "k2 tekerleği bulunamadı (CUDA 12.1 / torch 2.4.0 / $PYTAG)"; exit 1; }
pip install "$K2_URL"
pip install lhotse sentencepiece kaldifeat tensorboard "huggingface_hub[cli]" datasets soundfile \
            soxr onnx onnxruntime onnxsim

if [ ! -d icefall ]; then
  git clone https://github.com/k2-fsa/icefall.git
fi
git -C icefall fetch -q origin "$ICEFALL_COMMIT" && git -C icefall checkout -q "$ICEFALL_COMMIT"
pip install -r icefall/requirements.txt

# Temel model: Zipformer-68M'in kaynağı (VietASR, 70 bin saat; Apache-2.0)
huggingface-cli download zzasdf/viet_iter3_pseudo_label \
    --include "exp/epoch-12-avg-8.pt" "data/Vietnam_bpe_2000_new/*" --local-dir temel_model

python - <<'PY'
import k2, lhotse, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| k2", k2.__version__, "| lhotse", lhotse.__version__)
PY
echo "Hazır. Her oturumda: source $ROOT/venv/bin/activate && export PYTHONPATH=$ROOT/icefall:\$PYTHONPATH"
