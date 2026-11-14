#!/bin/bash
cd "$(dirname "$0")"
if [[ ! -x .venv_v42/bin/python ]]; then
  echo "[FAILED] Нет .venv_v42. Сначала должен быть установлен Python 3.12 и зависимости."
  exit 31
fi
.venv_v42/bin/python -c "import numpy,scipy,jax,matplotlib,PIL,imageio;assert numpy.__version__.startswith('2.');assert jax.__version__=='0.9.0.1'" || {
  echo "[FAILED] В окружении нет нужных библиотек."
  exit 31
}
echo "Очередь D: короткий предтест, затем 9 расчётов. Сводка будет обновляться на месте."
echo "Архив в конце: results_D/Send_to_AZ.zip"
.venv_v42/bin/python -u campaign_cd.py --participant D --test || exit 32
.venv_v42/bin/python -u campaign_cd.py --participant D
exit $?
