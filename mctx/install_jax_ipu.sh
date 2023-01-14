git clone https://github.com/google/jax --depth 1
cd jax/
sudo apt install g++ python python3-dev
pip install numpy wheel
git clone https://github.com/graphcore/tensorflow --branch '2.6/sdk-release-3.1' --depth 1
python build/build.py --bazel_options=--override_repository=org_tensorflow=./tensorflow
# pip install dist/*.whl
