# git clone https://github.com/google/jax --depth 1
cd jax/
# sudo apt install g++ python python3-dev
# pip install numpy wheel
# git clone https://github.com/graphcore/tensorflow --branch 'r2.6/sdk-release-2.6' --depth 1
python build/build.py --bazel_options=--override_repository=org_tensorflow=${PWD}/tensorflow
# pip install dist/*.whl
git clone https://github.com/graphcore/upstream-patches-awf-google-jax
