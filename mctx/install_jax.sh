git clone https://github.com/google/jax
cd jax/
sudo apt install g++ python python3-dev
pip install numpy wheel
python build/build.py
# python build/build.py --bazel_options=--override_repository=org_tensorflow=https://github.com/graphcore/tensorflow
pip install dist/*.whl
