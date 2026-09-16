"""PPO runner configuration for the Microtaur velocity task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def microtaur_velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),

            # Start with enough action noise to discover a gait before PPO
            # gradually narrows the policy distribution.
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),

        # The critic predicts state value only and does not need an action
        # distribution.
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),

        algorithm=RslRlPpoAlgorithmCfg(
            # Keep a small entropy bonus so exploration does not collapse
            # before a stable gait is established.
            entropy_coef=0.005,

            learning_rate=1e-3,
            schedule="adaptive",
            desired_kl=0.01,
            clip_param=0.2,
            num_learning_epochs=5,
            num_mini_batches=4,
            gamma=0.99,
            lam=0.95,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
        ),

        # 32-step rollouts give PPO enough temporal context to observe foot
        # contacts as well as the slower response to yaw commands.
        num_steps_per_env=32,
        experiment_name="microtaur_minitaur_velocity",
        max_iterations=10_000,
        save_interval=50,
    )