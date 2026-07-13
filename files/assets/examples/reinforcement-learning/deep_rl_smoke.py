"""Small deep-RL optimizer smoke checks and testable NumPy objectives."""

import argparse
from contextlib import contextmanager
from functools import wraps
import math
from numbers import Real
import random

import numpy as np


MIN_STEPS = 128
SAC_TARGET_ENTROPY = -1.0


def _numeric_vector(name, values):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{name} must contain real integer or floating values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(float, copy=True)


def _boolean_vector(name, values):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must contain booleans")
    return array.copy()


def _unit_interval(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def _same_length(*arrays):
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all transition arrays must have exactly the same shape")


def dqn_targets(rewards, next_q, terminated, gamma):
    """Return terminal-masked one-step DQN targets without broadcasting."""
    rewards = _numeric_vector("rewards", rewards)
    next_q = _numeric_vector("next_q", next_q)
    terminated = _boolean_vector("terminated", terminated)
    _same_length(rewards, next_q, terminated)
    gamma = _unit_interval("gamma", gamma)
    return rewards + gamma * (~terminated) * next_q


def discounted_returns(rewards, terminated, gamma):
    """Compute return-to-go, resetting the recurrence at true terminals."""
    rewards = _numeric_vector("rewards", rewards)
    terminated = _boolean_vector("terminated", terminated)
    _same_length(rewards, terminated)
    gamma = _unit_interval("gamma", gamma)
    returns = np.empty_like(rewards)
    running = 0.0
    for index in range(rewards.size - 1, -1, -1):
        running = rewards[index] + gamma * (not terminated[index]) * running
        returns[index] = running
    return returns


def gae(rewards, values, terminated, gamma, gae_lambda):
    """Compute generalized advantage estimates with one bootstrap value."""
    rewards = _numeric_vector("rewards", rewards)
    values = _numeric_vector("values", values)
    terminated = _boolean_vector("terminated", terminated)
    _same_length(rewards, terminated)
    if values.size != rewards.size + 1:
        raise ValueError("values must contain one value per step plus a bootstrap")
    gamma = _unit_interval("gamma", gamma)
    gae_lambda = _unit_interval("gae_lambda", gae_lambda)
    advantages = np.empty_like(rewards)
    running = 0.0
    for index in range(rewards.size - 1, -1, -1):
        continuing = not terminated[index]
        delta = (
            rewards[index]
            + gamma * continuing * values[index + 1]
            - values[index]
        )
        running = delta + gamma * gae_lambda * continuing * running
        advantages[index] = running
    return advantages


def ppo_clipped_loss(new_logp, old_logp, advantages, epsilon):
    """Return the scalar negative PPO clipped surrogate objective."""
    new_logp = _numeric_vector("new_logp", new_logp)
    old_logp = _numeric_vector("old_logp", old_logp)
    advantages = _numeric_vector("advantages", advantages)
    _same_length(new_logp, old_logp, advantages)
    epsilon = _unit_interval("epsilon", epsilon)
    if epsilon == 0.0 or epsilon == 1.0:
        raise ValueError("epsilon must be strictly between 0 and 1")
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.exp(new_logp - old_logp)
    if not np.all(np.isfinite(ratio)):
        raise ValueError("new_logp - old_logp produced a non-finite ratio")
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return float(-np.mean(np.minimum(unclipped, clipped)))


def sac_targets(
    rewards, next_min_q, next_logp, terminated, gamma, alpha
):
    """Return entropy-regularized, terminal-masked SAC critic targets."""
    rewards = _numeric_vector("rewards", rewards)
    next_min_q = _numeric_vector("next_min_q", next_min_q)
    next_logp = _numeric_vector("next_logp", next_logp)
    terminated = _boolean_vector("terminated", terminated)
    _same_length(rewards, next_min_q, next_logp, terminated)
    gamma = _unit_interval("gamma", gamma)
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite nonnegative number")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite nonnegative number")
    return rewards + gamma * (~terminated) * (next_min_q - alpha * next_logp)


def interaction_summary(
    interaction_count,
    total_reward,
    completed_episode_returns,
    current_partial_episode_return,
):
    """Summarize complete and partial episodes without inventing a mean."""
    if (
        isinstance(interaction_count, (bool, np.bool_))
        or not isinstance(interaction_count, (int, np.integer))
        or interaction_count <= 0
    ):
        raise ValueError("interaction_count must be a positive integer")
    scalars = {
        "total_reward": total_reward,
        "current_partial_episode_return": current_partial_episode_return,
    }
    for name, value in scalars.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number")
    completed = np.asarray(completed_episode_returns)
    if completed.ndim != 1:
        raise ValueError("completed_episode_returns must be one-dimensional")
    if completed.size:
        if not (
            np.issubdtype(completed.dtype, np.integer)
            or np.issubdtype(completed.dtype, np.floating)
        ) or not np.all(np.isfinite(completed)):
            raise ValueError("completed_episode_returns must contain finite real values")
        completed_mean = float(np.mean(completed))
    else:
        completed_mean = None
    return {
        "interaction_count": int(interaction_count),
        "reward_per_interaction": float(total_reward) / int(interaction_count),
        "completed_episode_count": int(completed.size),
        "completed_episode_return_mean": completed_mean,
        "current_partial_episode_return": float(current_partial_episode_return),
    }


def _print_run_summary(
    label,
    environment_id,
    interaction_count,
    total_reward,
    completed_episode_returns,
    current_partial_episode_return,
    metrics,
):
    summary = interaction_summary(
        interaction_count,
        total_reward,
        completed_episode_returns,
        current_partial_episode_return,
    )
    fields = [f"env={environment_id}"]
    for name, value in summary.items():
        if value is None:
            rendered = "unavailable"
        elif isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        fields.append(f"{name}={rendered}")
    for name, value in metrics.items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        fields.append(f"{name}={rendered}")
    print(f"{label} summary: " + " ".join(fields))


@contextmanager
def managed_environment(environment):
    """Close a Gymnasium-like environment even when the run raises."""
    try:
        yield environment
    finally:
        environment.close()


def _with_environment(environment_id):
    def decorate(function):
        @wraps(function)
        def wrapped(seed, steps):
            import gymnasium as gym

            with managed_environment(gym.make(environment_id)) as environment:
                return function(seed, steps, environment)

        return wrapped

    return decorate


def torch_dqn_target_and_loss(predicted_q, rewards, next_q, terminated, gamma):
    """Build the detached DQN target and its Huber loss."""
    import torch
    from torch.nn import functional as F

    with torch.no_grad():
        targets = rewards + gamma * (~terminated).to(rewards.dtype) * next_q
    return targets, F.smooth_l1_loss(predicted_q, targets)


def torch_ppo_minibatch_losses(
    actor,
    critic,
    states,
    actions,
    old_logp,
    advantages,
    returns,
    clip_epsilon=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
):
    """Compute the four PPO minibatch terms used by the optimizer."""
    import torch
    from torch.nn import functional as F

    distribution = torch.distributions.Categorical(logits=actor(states))
    new_logp = distribution.log_prob(actions)
    ratio = torch.exp(new_logp - old_logp)
    surrogate = torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
    )
    policy_loss = -surrogate.mean()
    value_loss = F.mse_loss(critic(states).squeeze(1), returns)
    entropy = distribution.entropy().mean()
    combined_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "combined_loss": combined_loss,
        "new_logp": new_logp,
    }


def torch_sampled_approximate_kl(old_logp, final_logp):
    """Return sampled mean old_logp - final_logp; it may be negative."""
    return (old_logp.detach() - final_logp.detach()).mean()


def torch_sac_targets(rewards, next_min_q, next_logp, terminated, gamma, alpha):
    """Build the detached entropy-regularized SAC critic target."""
    import torch

    with torch.no_grad():
        return rewards + gamma * (~terminated).to(rewards.dtype) * (
            next_min_q - alpha * next_logp
        )


def torch_temperature_loss(log_alpha, logp, target_entropy):
    """Return the automatic-temperature objective with detached policy logp."""
    return -(log_alpha * (logp.detach() + target_entropy)).mean()


def polyak_update(target_parameters, source_parameters, tau):
    """Move target parameters by tau toward source parameters."""
    import torch

    if len(target_parameters) != len(source_parameters):
        raise ValueError("target and source parameter lists must have equal length")
    tau = _unit_interval("tau", tau)
    with torch.no_grad():
        for target, source in zip(target_parameters, source_parameters):
            target.mul_(1.0 - tau).add_(source, alpha=tau)


@contextmanager
def frozen_parameters(parameters):
    """Temporarily freeze a cached parameter list while preserving action grads."""
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous):
            parameter.requires_grad_(requires_grad)


def _seed_everything(seed, torch, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    env.action_space.seed(seed)


def _parameter_delta(before, parameters, torch):
    return float(
        sum(
            torch.sum(torch.abs(old - new.detach())).item()
            for old, new in zip(before, parameters)
        )
    )


def _assert_finite_update(label, metrics, parameter_delta):
    if parameter_delta <= 0.0 or not all(math.isfinite(x) for x in metrics):
        raise RuntimeError(f"{label} did not produce a finite parameter update")


@_with_environment("CartPole-v1")
def run_dqn(seed, steps, env):
    import torch
    from torch import nn

    _seed_everything(seed, torch, env)
    rng = np.random.default_rng(seed)
    online = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in online.parameters()]
    replay = []
    batch_size = 64
    losses = []
    episode_returns = []
    episode_return = 0.0
    total_reward = 0.0
    observation, _ = env.reset(seed=seed)

    for step in range(steps):
        epsilon = max(0.1, 1.0 - step / steps)
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(
                    online(torch.as_tensor(observation, dtype=torch.float32)).argmax()
                )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay.append(
            (
                observation.copy(),
                action,
                reward,
                next_observation.copy(),
                terminated,
            )
        )
        episode_return += reward
        total_reward += reward
        observation = next_observation
        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset()

        if len(replay) < batch_size:
            continue
        indices = rng.choice(len(replay), size=batch_size, replace=False)
        batch = [replay[index] for index in indices]
        states = torch.as_tensor(np.stack([row[0] for row in batch]), dtype=torch.float32)
        actions = torch.as_tensor([row[1] for row in batch], dtype=torch.int64)
        rewards = torch.as_tensor([row[2] for row in batch], dtype=torch.float32)
        next_states = torch.as_tensor(
            np.stack([row[3] for row in batch]), dtype=torch.float32
        )
        terminals = torch.as_tensor([row[4] for row in batch], dtype=torch.bool)
        predicted = online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = target(next_states).max(dim=1).values
        _, loss = torch_dqn_target_and_loss(
            predicted, rewards, next_q, terminals, gamma=0.99
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if len(losses) % 32 == 0:
            target.load_state_dict(online.state_dict())

    delta = _parameter_delta(before, online.parameters(), torch)
    _assert_finite_update("DQN", [losses[-1], delta], delta)
    _print_run_summary(
        "DQN",
        "CartPole-v1",
        steps,
        total_reward,
        episode_returns,
        episode_return,
        {
            "updates": len(losses),
            "loss": losses[-1],
            "parameter_delta": delta,
        },
    )


@_with_environment("CartPole-v1")
def run_reinforce(seed, steps, env):
    import torch
    from torch import nn

    _seed_everything(seed, torch, env)
    policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    all_logp = []
    all_returns = []
    episode_returns = []
    collected = 0

    while collected < steps:
        observation, _ = env.reset(seed=seed if collected == 0 else None)
        rewards = []
        logps = []
        terminated_flags = []
        done = False
        while not done:
            logits = policy(torch.as_tensor(observation, dtype=torch.float32))
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            observation, reward, terminated, truncated, _ = env.step(int(action))
            logps.append(distribution.log_prob(action))
            rewards.append(reward)
            terminated_flags.append(terminated)
            collected += 1
            done = terminated or truncated
        # A truncation ends this sampled trajectory. With no critic, its omitted
        # tail is not bootstrapped, so this finite-horizon estimator can be biased.
        returns = discounted_returns(
            np.asarray(rewards), np.asarray(terminated_flags), gamma=0.99
        )
        all_logp.extend(logps)
        all_returns.extend(returns.tolist())
        episode_returns.append(float(sum(rewards)))

    returns_tensor = torch.as_tensor(all_returns, dtype=torch.float32)
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (
        returns_tensor.std(unbiased=False) + 1e-8
    )
    loss = -(torch.stack(all_logp) * returns_tensor).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    delta = _parameter_delta(before, policy.parameters(), torch)
    loss_value = float(loss.detach())
    total_reward = float(sum(episode_returns))
    metrics = [loss_value, total_reward, delta]
    _assert_finite_update("REINFORCE", metrics, delta)
    _print_run_summary(
        "REINFORCE",
        "CartPole-v1",
        collected,
        total_reward,
        episode_returns,
        0.0,
        {
            "requested_steps": steps,
            "loss": loss_value,
            "parameter_delta": delta,
        },
    )


@_with_environment("CartPole-v1")
def run_ppo(seed, steps, env):
    import torch
    from torch import nn

    _seed_everything(seed, torch, env)
    actor = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
    critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-3
    )
    parameters = list(actor.parameters()) + list(critic.parameters())
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    segments = []
    segment = []
    episode_returns = []
    episode_return = 0.0
    total_reward = 0.0
    observation, _ = env.reset(seed=seed)

    for index in range(steps):
        state_tensor = torch.as_tensor(observation, dtype=torch.float32)
        distribution = torch.distributions.Categorical(logits=actor(state_tensor))
        action = distribution.sample()
        value = critic(state_tensor).squeeze()
        next_observation, reward, terminated, truncated, _ = env.step(int(action))
        segment.append(
            (
                observation.copy(),
                int(action),
                float(distribution.log_prob(action).detach()),
                float(value.detach()),
                reward,
                terminated,
            )
        )
        episode_return += reward
        total_reward += reward
        observation = next_observation
        rollout_ended = index == steps - 1
        if terminated or truncated or rollout_ended:
            if terminated:
                bootstrap = 0.0
            else:
                with torch.no_grad():
                    bootstrap = float(
                        critic(
                            torch.as_tensor(next_observation, dtype=torch.float32)
                        ).squeeze()
                    )
            segments.append((segment, bootstrap))
            segment = []
        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset()

    states_rows, action_rows, old_logp_rows = [], [], []
    old_value_rows, advantage_rows = [], []
    for rows, bootstrap in segments:
        rewards = np.asarray([row[4] for row in rows])
        terminals = np.asarray([row[5] for row in rows], dtype=bool)
        values = np.asarray([row[3] for row in rows] + [bootstrap])
        advantages = gae(rewards, values, terminals, 0.99, 0.95)
        states_rows.extend(row[0] for row in rows)
        action_rows.extend(row[1] for row in rows)
        old_logp_rows.extend(row[2] for row in rows)
        old_value_rows.extend(row[3] for row in rows)
        advantage_rows.extend(advantages.tolist())

    states = torch.as_tensor(np.stack(states_rows), dtype=torch.float32)
    actions = torch.as_tensor(action_rows, dtype=torch.int64)
    old_logp = torch.as_tensor(old_logp_rows, dtype=torch.float32).detach()
    old_values = torch.as_tensor(old_value_rows, dtype=torch.float32)
    advantages = torch.as_tensor(advantage_rows, dtype=torch.float32)
    returns = advantages + old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    losses = []
    for indices in torch.randperm(steps).split(64):
        terms = torch_ppo_minibatch_losses(
            actor,
            critic,
            states[indices],
            actions[indices],
            old_logp[indices],
            advantages[indices],
            returns[indices],
        )
        optimizer.zero_grad()
        terms["combined_loss"].backward()
        nn.utils.clip_grad_norm_(parameters, 0.5)
        optimizer.step()
        losses.append(float(terms["combined_loss"].detach()))

    with torch.no_grad():
        final_distribution = torch.distributions.Categorical(logits=actor(states))
        final_logp = final_distribution.log_prob(actions)
        sampled_approximate_kl = float(
            torch_sampled_approximate_kl(old_logp, final_logp)
        )
    actor_delta = _parameter_delta(actor_before, actor.parameters(), torch)
    critic_delta = _parameter_delta(critic_before, critic.parameters(), torch)
    _assert_finite_update(
        "PPO actor", [losses[-1], sampled_approximate_kl, actor_delta], actor_delta
    )
    _assert_finite_update(
        "PPO critic", [losses[-1], sampled_approximate_kl, critic_delta], critic_delta
    )
    _print_run_summary(
        "PPO",
        "CartPole-v1",
        steps,
        total_reward,
        episode_returns,
        episode_return,
        {
            "minibatches": len(losses),
            "loss": losses[-1],
            "sampled_approximate_kl": sampled_approximate_kl,
            "actor_parameter_delta": actor_delta,
            "critic_parameter_delta": critic_delta,
        },
    )


@_with_environment("Pendulum-v1")
def run_sac(seed, steps, env):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Actor(nn.Module):
        def __init__(self, low, high):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.mean = nn.Linear(64, 1)
            self.log_std = nn.Linear(64, 1)
            self.register_buffer("scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
            self.register_buffer("bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

        def sample(self, states):
            hidden = self.body(states)
            mean = self.mean(hidden)
            log_std = self.log_std(hidden).clamp(-5.0, 2.0)
            normal = torch.distributions.Normal(mean, log_std.exp())
            raw_action = normal.rsample()
            normalized_action = torch.tanh(raw_action)
            action = normalized_action * self.scale + self.bias
            logp = normal.log_prob(raw_action)
            logp -= torch.log(1.0 - normalized_action.pow(2) + 1e-6)
            logp -= torch.log(self.scale)
            return action, logp.sum(dim=-1)

    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        def forward(self, states, actions):
            return self.network(torch.cat((states, actions), dim=-1)).squeeze(-1)

    _seed_everything(seed, torch, env)
    rng = np.random.default_rng(seed)
    actor = Actor(env.action_space.low, env.action_space.high)
    q1, q2, target_q1, target_q2 = QNetwork(), QNetwork(), QNetwork(), QNetwork()
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    critic_parameters = list(q1.parameters()) + list(q2.parameters())
    target_q1_parameters = list(target_q1.parameters())
    target_q2_parameters = list(target_q2.parameters())
    q1_parameters = list(q1.parameters())
    q2_parameters = list(q2.parameters())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=3e-4)
    log_alpha = torch.tensor(math.log(0.2), requires_grad=True)
    target_entropy = SAC_TARGET_ENTROPY
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in q1.parameters()]
    alpha_before = float(log_alpha.detach())
    replay = []
    batch_size = 64
    losses = []
    episode_returns = []
    episode_return = 0.0
    total_reward = 0.0
    observation, _ = env.reset(seed=seed)

    for step in range(steps):
        if step < batch_size:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor.sample(
                    torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                )[0].squeeze(0).numpy()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay.append(
            (observation.copy(), action.copy(), reward, next_observation.copy(), terminated)
        )
        episode_return += reward
        total_reward += reward
        observation = next_observation
        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset()
        if len(replay) < batch_size:
            continue

        indices = rng.choice(len(replay), size=batch_size, replace=False)
        batch = [replay[index] for index in indices]
        states = torch.as_tensor(np.stack([row[0] for row in batch]), dtype=torch.float32)
        actions = torch.as_tensor(np.stack([row[1] for row in batch]), dtype=torch.float32)
        rewards = torch.as_tensor([row[2] for row in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([row[3] for row in batch]), dtype=torch.float32)
        terminals = torch.as_tensor([row[4] for row in batch], dtype=torch.bool)
        with torch.no_grad():
            next_actions, next_logp = actor.sample(next_states)
            next_min_q = torch.minimum(
                target_q1(next_states, next_actions), target_q2(next_states, next_actions)
            )
        targets = torch_sac_targets(
            rewards,
            next_min_q,
            next_logp,
            terminals,
            gamma=0.99,
            alpha=log_alpha.exp(),
        )
        critic_loss = F.mse_loss(q1(states, actions), targets) + F.mse_loss(
            q2(states, actions), targets
        )
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        with frozen_parameters(critic_parameters):
            sampled_actions, logp = actor.sample(states)
            actor_loss = (
                log_alpha.exp().detach() * logp
                - torch.minimum(
                    q1(states, sampled_actions), q2(states, sampled_actions)
                )
            ).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

        alpha_loss = torch_temperature_loss(log_alpha, logp, target_entropy)
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        polyak_update(target_q1_parameters, q1_parameters, tau=0.005)
        polyak_update(target_q2_parameters, q2_parameters, tau=0.005)
        losses.append(
            (float(critic_loss.detach()), float(actor_loss.detach()), float(alpha_loss.detach()))
        )

    actor_delta = _parameter_delta(actor_before, actor.parameters(), torch)
    critic_delta = _parameter_delta(critic_before, q1.parameters(), torch)
    alpha_delta = abs(float(log_alpha.detach()) - alpha_before)
    critic_loss, actor_loss, alpha_loss = losses[-1]
    metrics = [critic_loss, actor_loss, alpha_loss, actor_delta, critic_delta, alpha_delta]
    _assert_finite_update("SAC actor", metrics, actor_delta)
    _assert_finite_update("SAC critic", metrics, critic_delta)
    if alpha_delta <= 0.0:
        raise RuntimeError("SAC temperature did not update")
    _print_run_summary(
        "SAC",
        "Pendulum-v1",
        steps,
        total_reward,
        episode_returns,
        episode_return,
        {
            "updates": len(losses),
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha": float(log_alpha.exp().detach()),
            "actor_parameter_delta": actor_delta,
            "critic_parameter_delta": critic_delta,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run one real deep-RL optimizer smoke update (not convergence training)."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("dqn", "reinforce", "ppo", "sac"):
        subparser = subparsers.add_parser(mode)
        subparser.add_argument("--seed", type=int, default=7)
        subparser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    if args.steps < MIN_STEPS:
        parser.error(f"--steps must be at least {MIN_STEPS} for every smoke mode")
    dispatch = {
        "dqn": run_dqn,
        "reinforce": run_reinforce,
        "ppo": run_ppo,
        "sac": run_sac,
    }
    dispatch[args.mode](args.seed, args.steps)


if __name__ == "__main__":
    main()
