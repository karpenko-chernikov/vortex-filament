#!/bin/bash
cd "$(dirname "$0")"
if command -v python3.12 >/dev/null 2>&1; then
  PY312="$(command -v python3.12)"
elif [[ -x /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 ]]; then
  PY312="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
else
  echo "[FAILED] Нужен Python 3.12."
  exit 31
fi
if [[ ! -x .venv_v42/bin/python ]]; then
  "$PY312" -m venv .venv_v42 || exit 31
fi
check_packages() {
  .venv_v42/bin/python -c "import numpy,scipy,jax,matplotlib,PIL,imageio;assert numpy.__version__.startswith('2.');assert jax.__version__=='0.9.0.1'"
}
if ! check_packages; then
  .venv_v42/bin/python -m pip install --upgrade pip || exit 31
  .venv_v42/bin/python -m pip install -r requirements.txt || exit 31
  check_packages || exit 31
fi
echo "Короткий предтест, затем очередь участника A."
echo "Сводка прогресса — примерно каждые 20 секунд. Архив: results_A/Send_to_AZ.zip"
.venv_v42/bin/python -u distributed_runner.py --participant A --test || exit 32
.venv_v42/bin/python -u distributed_runner.py --participant A
exit $?
