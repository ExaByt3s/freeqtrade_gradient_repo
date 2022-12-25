import numpy as np

# import srl
import simple_distributed_rl.srl as srl

from srl import runner
from srl.rl.models.alphazero_image_block import AlphaZeroImageBlock

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import stochastic_muzero  # isort: skip

import gym
from freqtrade.freqai.RL.Base5ActionRLEnv import Base5ActionRLEnv

gym.envs.registration.register(
    id='Base5ActionRLEnv-v0',
    entry_point=__name__ + ':Base5ActionRLEnv',
    max_episode_steps=10,
)

def main():
    rl_config = stochastic_muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=16,
        memory_warmup_size=200,
        memory_name="ReplayMemory",
        lr_init=0.01,
        lr_decay_steps=10_000,
        v_min=-2,
        v_max=2,
        unroll_steps=1,
        input_image_block=AlphaZeroImageBlock,
        input_image_block_kwargs={"n_blocks": 1, "filters": 16},
        dynamics_blocks=1,
        enable_rescale=False,
        codebook_size=4,
    )
    rl_config.processors = [grid.LayerProcessor()]
    # env_config = srl.EnvConfig("Grid")
    env_config = gym.make('Base5ActionRLEnv-v0')

    config = runner.Config(env_config, rl_config)

    config.model_summary()

    # --- 学習ループ
    parameter, memory, history = runner.train(config, max_episodes=1)
    # history.plot(plot_right=["train_loss", "train_policy_loss"])
    print(history)

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=10, print_progress=True)
    print("mean", np.mean(rewards))

    # --- rendering
    # rewards, render = runner.render(config, parameter, print_progress=True)
    # render.create_anime(interval=1000 / 2, scale=2).save("_qiita.gif")


if __name__ == "__main__":
    main()
