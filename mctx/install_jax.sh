git clone https://github.com/google/jax
cd jax/
sudo apt install g++ python python3-dev
pip install numpy wheel
python build/build.py
pip install dist/*.whl
