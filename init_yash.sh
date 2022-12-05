export EDITOR='micro'
export TF_POPLAR_FLAGS="--executable_cache_path=/tmp/poplar_exec --max_compilation_threads=$(nproc) --max_infeed_threads=$(nproc)"

alias e="${EDITOR}"
alias edit="${EDITOR}"
unset COLUMNS
