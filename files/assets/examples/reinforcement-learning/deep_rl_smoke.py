"""Small deep-RL optimizer smoke checks and testable NumPy objectives."""

import argparse
import math
import random

import numpy as np


MIN_STEPS = 128


def _numeric_vector(name, values):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must contain numeric values")
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
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
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(
        alpha, (int, float, np.number)
    ):
        raise ValueError("alpha must be a finite nonnegative number")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite nonnegative number")
    return rewards + gamma * (~terminated) * (next_min_q - alpha * next_logp)


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


def run_dqn(seed, steps):
    import gymnasium as gym
    import torch
    from torch import nn
    from torch.nn import functional as F

    env = gym.make("CartPole-v1")
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
        terminals = torch.as_tensor([row[4] for row in batch], dtype=torch.float32)
        predicted = online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = target(next_states).max(dim=1).values
            targets = rewards + 0.99 * (1.0 - terminals) * next_q
        loss = F.smooth_l1_loss(predicted, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if len(losses) % 32 == 0:
            target.load_state_dict(online.state_dict())

    env.close()
    delta = _parameter_delta(before, online.parameters(), torch)
    _assert_finite_update("DQN", [losses[-1], delta], delta)
    mean_return = float(np.mean(episode_returns or [episode_return]))
    print(
        "DQN summary: env=CartPole-v1 "
        f"steps={steps} updates={len(losses)} loss={losses[-1]:.6f} "
        f"episode_return_mean={mean_return:.3f} parameter_delta={delta:.6f}"
    )


def run_reinforce(seed, steps):
    import gymnasium as gym
    import torch
    from torch import nn

    env = gym.make("CartPole-v1")
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
    env.close()
    delta = _parameter_delta(before, policy.parameters(), torch)
    metrics = [float(loss.detach()), float(np.mean(episode_returns)), delta]
    _assert_finite_update("REINFORCE", metrics, delta)
    print(
        "REINFORCE summary: env=CartPole-v1 "
        f"requested_steps={steps} collected_steps={collected} "
        f"episodes={len(episode_returns)} loss={metrics[0]:.6f} "
        f"episode_return_mean={metrics[1]:.3f} parameter_delta={delta:.6f}"
    )


def run_ppo(seed, steps):
    import gymnasium as gym
    import torch
    from torch import nn
    from torch.nn import functional as F

    env = gym.make("CartPole-v1")
    _seed_everything(seed, torch, env)
    actor = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
    critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-3
    )
    parameters = list(actor.parameters()) + list(critic.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    segments = []
    segment = []
    episode_returns = []
    episode_return = 0.0
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
    approximate_kl = 0.0
    for indices in torch.randperm(steps).split(64):
        distribution = torch.distributions.Categorical(logits=actor(states[indices]))
        new_logp = distribution.log_prob(actions[indices])
        ratio = torch.exp(new_logp - old_logp[indices])
        surrogate = torch.minimum(
            ratio * advantages[indices],
            torch.clamp(ratio, 0.8, 1.2) * advantages[indices],
        )
        value_loss = F.mse_loss(critic(states[indices]).squeeze(1), returns[indices])
        entropy = distribution.entropy().mean()
        loss = -surrogate.mean() + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 0.5)
        optimizer.step()
        losses.append(float(loss.detach()))
        approximate_kl = float((old_logp[indices] - new_logp.detach()).mean())

    env.close()
    delta = _parameter_delta(before, parameters, torch)
    mean_return = float(np.mean(episode_returns or [episode_return]))
    _assert_finite_update(
        "PPO", [losses[-1], approximate_kl, mean_return, delta], delta
    )
    print(
        "PPO summary: env=CartPole-v1 "
        f"steps={steps} minibatches={len(losses)} loss={losses[-1]:.6f} "
        f"approx_kl={approximate_kl:.6f} episode_return_mean={mean_return:.3f} "
        f"parameter_delta={delta:.6f}"
    )


def run_sac(seed, steps):
    import gymnasium as gym
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

    env = gym.make("Pendulum-v1")
    _seed_everything(seed, torch, env)
    rng = np.random.default_rng(seed)
    actor = Actor(env.action_space.low, env.action_space.high)
    q1, q2, target_q1, target_q2 = QNetwork(), QNetwork(), QNetwork(), QNetwork()
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=3e-4
    )
    log_alpha = torch.tensor(math.log(0.2), requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in q1.parameters()]
    alpha_before = float(log_alpha.detach())
    replay = []
    batch_size = 64
    losses = []
    episode_returns = []
    episode_return = 0.0
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
        terminals = torch.as_tensor([row[4] for row in batch], dtype=torch.float32)
        with torch.no_grad():
            next_actions, next_logp = actor.sample(next_states)
            next_min_q = torch.minimum(
                target_q1(next_states, next_actions), target_q2(next_states, next_actions)
            )
            targets = rewards + 0.99 * (1.0 - terminals) * (
                next_min_q - log_alpha.exp() * next_logp
            )
        critic_loss = F.mse_loss(q1(states, actions), targets) + F.mse_loss(
            q2(states, actions), targets
        )
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        for parameter in list(q1.parameters()) + list(q2.parameters()):
            parameter.requires_grad_(False)
        sampled_actions, logp = actor.sample(states)
        actor_loss = (log_alpha.exp().detach() * logp - torch.minimum(
            q1(states, sampled_actions), q2(states, sampled_actions)
        )).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        for parameter in list(q1.parameters()) + list(q2.parameters()):
            parameter.requires_grad_(True)

        alpha_loss = -(log_alpha * (logp.detach() - 1.0)).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(target_q1.parameters(), q1.parameters()):
                target_parameter.mul_(0.995).add_(parameter, alpha=0.005)
            for target_parameter, parameter in zip(target_q2.parameters(), q2.parameters()):
                target_parameter.mul_(0.995).add_(parameter, alpha=0.005)
        losses.append(
            (float(critic_loss.detach()), float(actor_loss.detach()), float(alpha_loss.detach()))
        )

    env.close()
    actor_delta = _parameter_delta(actor_before, actor.parameters(), torch)
    critic_delta = _parameter_delta(critic_before, q1.parameters(), torch)
    alpha_delta = abs(float(log_alpha.detach()) - alpha_before)
    critic_loss, actor_loss, alpha_loss = losses[-1]
    metrics = [critic_loss, actor_loss, alpha_loss, actor_delta, critic_delta, alpha_delta]
    _assert_finite_update("SAC actor", metrics, actor_delta)
    _assert_finite_update("SAC critic", metrics, critic_delta)
    if alpha_delta <= 0.0:
        raise RuntimeError("SAC temperature did not update")
    mean_return = float(np.mean(episode_returns or [episode_return]))
    print(
        "SAC summary: env=Pendulum-v1 "
        f"steps={steps} updates={len(losses)} critic_loss={critic_loss:.6f} "
        f"actor_loss={actor_loss:.6f} alpha={float(log_alpha.exp().detach()):.6f} "
        f"episode_return_mean={mean_return:.3f} actor_delta={actor_delta:.6f} "
        f"critic_delta={critic_delta:.6f}"
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
