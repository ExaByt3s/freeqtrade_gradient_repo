#!/usr/bin/yash

# POPLAR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POPLAR_ROOT='/opt/poplar'

if echo "${POPLAR_SDK_ENABLED:+:${POPLAR_SDK_ENABLED}}" | grep --invert-match "${POPLAR_ROOT}"; then
  export CMAKE_PREFIX_PATH=${POPLAR_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}
  export PATH=${POPLAR_ROOT}/bin${PATH:+:${PATH}}
  export CPATH=${POPLAR_ROOT}/include${CPATH:+:${CPATH}}
  export LIBRARY_PATH=${POPLAR_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
  export LD_LIBRARY_PATH=${POPLAR_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  export OPAL_PREFIX=${POPLAR_ROOT}
  export PYTHONPATH=${POPLAR_ROOT}/python:${POPLAR_ROOT}/lib/python${PYTHONPATH:+:${PYTHONPATH}}
  export POPLAR_SDK_ENABLED=${POPLAR_ROOT}
else
  echo 'ERROR: A Poplar SDK has already been enabled.'
  echo "Path of enabled Poplar SDK: ${POPLAR_SDK_ENABLED}"
fi
