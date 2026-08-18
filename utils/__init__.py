"""RLStudy 共享工具包。

快速访问最常用的函数和类。
"""
from .seed import set_seed
from .viz import (
    LivePlot,
    animate_agent,
    plot_bar_compare,
    plot_q_table,
    plot_regret_curve,
    plot_training_curve,
    plot_value_heatmap,
    save_animation,
    smooth,
)
# ipywidgets 是软依赖：缺失时 make_interactive 自动降级为静态图，
# 不装它也不影响 import utils（README 的"静态降级"承诺由此兑现）。
_WIDGET_EXPORTS = [
    "ensure_widget_backend", "hbox", "make_checkbox", "make_dropdown",
    "make_interactive", "make_int_slider", "make_run_button", "make_slider",
    "vbox",
]
try:
    from .widgets import (  # noqa: F401
        ensure_widget_backend,
        hbox,
        make_checkbox,
        make_dropdown,
        make_interactive,
        make_int_slider,
        make_run_button,
        make_slider,
        vbox,
    )
except ImportError:
    pass
from .animation import make_frame_getter, save_gif, save_mp4

# torch_utils / networks / replay / dqn_utils 涉及 PyTorch，延迟导入避免 Phase 1 强依赖 torch
try:
    from .torch_utils import count_parameters, get_device, grad_stats  # noqa: F401
except ImportError:
    pass

try:
    from .networks import DuelingQNetwork, QNetwork, make_mlp  # noqa: F401
except ImportError:
    pass

try:
    from .replay import ReplayBuffer, Transition  # noqa: F401
except ImportError:
    pass

try:
    from .dqn_utils import (  # noqa: F401
        dqn_update_step,
        epsilon_greedy_action,
        hard_update,
        linear_epsilon_schedule,
        polyak_update,
    )
except ImportError:
    pass

try:
    from .policy_networks import CategoricalPolicy, GaussianPolicy  # noqa: F401
except ImportError:
    pass

try:
    from .policy_networks import ValueNetwork, ActorCritic  # noqa: F401
except ImportError:
    pass

try:
    from .gae import compute_gae, compute_td_errors, compute_n_step_advantage, compute_returns_from_gae  # noqa: F401
except ImportError:
    pass

try:
    from .ppo import (  # noqa: F401
        approx_kl_from_ratio,
        compute_clip_objective,
        compute_kl,
        ppo_update,
    )
except ImportError:
    pass

try:
    from .reward_model import (  # noqa: F401
        RewardModel,
        bradley_terry_loss,
        generate_preference_data,
        make_preference_batch,
        pad_to_length,
        predict_rewards,
        reward_accuracy,
        true_reward,
    )
except ImportError:
    pass

try:
    from .rlhf import (  # noqa: F401
        RLHFConfig,
        RLHFTrainer,
        ValueHead,
    )
except ImportError:
    pass

try:
    from .grpo import (  # noqa: F401
        GRPOConfig,
        GRPOTrainer,
        compute_group_advantages,
    )
except ImportError:
    pass

try:
    from .dpo import (  # noqa: F401
        DPOConfig,
        DPOTrainer,
        KTOTrainer,
        dpo_loss,
        kto_loss,
        kto_points_to_loss,
        prospect_value,
        sequence_log_probs,
    )
except ImportError:
    pass

try:
    from .prm import (  # noqa: F401
        ProcessRewardModel,
        encode_two_step_sample,
        evaluate_two_step_accuracy,
        make_two_step_addition_dataset,
        make_wrong_step_variations,
        orm_best_of_n,
        parse_two_step_response,
        prm_best_of_n,
        step_level_loss,
        step_rewards_from_token_rewards,
    )
except ImportError:
    pass

try:
    from .self_play import (  # noqa: F401
        AIJudge,
        Constitution,
        generate_ai_preferences,
        self_reward_score,
        spin_iteration,
        spin_objective,
    )
except ImportError:
    pass

try:
    from .offline_rl import (  # noqa: F401
        CQLTrainer,
        DTTrainer,
        DecisionTransformer,
        IQLTrainer,
        OfflineDataset,
        cql_loss,
        collect_offline_dataset,
        dt_loss,
        dt_rollout,
        evaluate_policy,
        expectile_loss,
        heuristic_cartpole_policy,
        offline_dqn_update_step,
        random_policy_factory,
    )
except ImportError:
    pass

# torch 依赖的名字：**导入成功多少就导出多少**——否则 `from utils import *`
# 会静默丢掉 RLHFTrainer / GRPOTrainer / DPOTrainer 等。
_TORCH_EXPORTS = [
    # torch_utils
    "count_parameters", "get_device", "grad_stats",
    # networks / replay / dqn_utils
    "DuelingQNetwork", "QNetwork", "make_mlp",
    "ReplayBuffer", "Transition",
    "dqn_update_step", "epsilon_greedy_action", "hard_update",
    "linear_epsilon_schedule", "polyak_update",
    # policy networks
    "CategoricalPolicy", "GaussianPolicy", "ValueNetwork", "ActorCritic",
    # gae / ppo
    "compute_gae", "compute_td_errors", "compute_n_step_advantage",
    "compute_returns_from_gae",
    "approx_kl_from_ratio", "compute_clip_objective", "compute_kl", "ppo_update",
    # reward model
    "RewardModel", "bradley_terry_loss", "generate_preference_data",
    "make_preference_batch", "pad_to_length", "predict_rewards",
    "reward_accuracy", "true_reward",
    # rlhf / grpo
    "RLHFConfig", "RLHFTrainer", "ValueHead",
    "GRPOConfig", "GRPOTrainer", "compute_group_advantages",
    # dpo / kto
    "DPOConfig", "DPOTrainer", "KTOTrainer", "dpo_loss", "kto_loss",
    "kto_points_to_loss", "prospect_value", "sequence_log_probs",
    # prm
    "ProcessRewardModel", "encode_two_step_sample",
    "evaluate_two_step_accuracy", "make_two_step_addition_dataset",
    "make_wrong_step_variations", "orm_best_of_n", "parse_two_step_response",
    "prm_best_of_n", "step_level_loss", "step_rewards_from_token_rewards",
    # self play
    "AIJudge", "Constitution", "generate_ai_preferences",
    "self_reward_score", "spin_iteration", "spin_objective",
    # offline rl
    "CQLTrainer", "DTTrainer", "DecisionTransformer", "IQLTrainer",
    "OfflineDataset", "cql_loss", "collect_offline_dataset", "dt_loss",
    "dt_rollout", "evaluate_policy", "expectile_loss",
    "heuristic_cartpole_policy", "offline_dqn_update_step",
    "random_policy_factory",
]

__all__ = [
    # seed
    "set_seed",
    # viz
    "LivePlot",
    "animate_agent",
    "plot_bar_compare",
    "plot_q_table",
    "plot_regret_curve",
    "plot_training_curve",
    "plot_value_heatmap",
    "save_animation",
    "smooth",
    # animation
    "make_frame_getter",
    "save_gif",
    "save_mp4",
] + [name for name in _WIDGET_EXPORTS if name in globals()] \
  + [name for name in _TORCH_EXPORTS if name in globals()]
