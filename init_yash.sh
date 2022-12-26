export EDITOR='micro'
export TF_POPLAR_FLAGS="--executable_cache_path=/tmp/poplar_exec --max_compilation_threads=$(nproc) --max_infeed_threads=$(nproc)"

alias e="${EDITOR}"
alias edit="${EDITOR}"
unset COLUMNS

pip install -U tensorflow

# . ./.venv/bin/activate
. ./freqtrade/freqtrade/.env/bin/activate

export OLDPYTHONPATH="${PYTHONPATH}"
export PYTHONPATH="${OLDPYTHONPATH}"
export PYTHONPATH="/usr/local/lib/python3.8/dist-packages:${PYTHONPATH}"
export PYTHONPATH="/notebooks/freqtrade/freqtrade/.env/lib/python3.8/site-packages:${PYTHONPATH}"

export COLORTERM='truecolor'
export XDG_CONFIG_HOME='/notebooks/setting'

mkdir /tmp/model
