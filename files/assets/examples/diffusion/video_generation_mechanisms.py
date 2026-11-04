"""Stdlib-only numerical illustrations; no video model or learned weights.

Noise time tau=0 is data, tau=1 is noise. Run this file directly.
The shape calculator uses Wan2.1's supported 4k+1 frame layout and (1,2,2)
latent patches. The Gaussian flow is an analytic 1-D distribution example.
"""
import json
import math


def _paired(a, b):
    if not a or len(a) != len(b):
        raise ValueError('expected nonempty equal-length vectors')


def video_budget(frames=81, height=480, width=832):
    if any(type(n) is not int or n <= 0 for n in (frames, height, width)):
        raise ValueError('dimensions must be positive integers')
    if (frames - 1) % 4 or height % 16 or width % 16:
        raise ValueError('requires F=4k+1 and H,W divisible by 16 for this VAE + patch layout')
    latent_frames, latent_h, latent_w = 1 + (frames - 1) // 4, height // 8, width // 8
    spatial = (latent_h // 2) * (latent_w // 2)
    tokens = latent_frames * spatial
    return dict(latent_shape=(16, latent_frames, latent_h, latent_w),
                latent_elements=16 * latent_frames * latent_h * latent_w,
                patch_features=16 * 1 * 2 * 2, tokens=tokens,
                dense_pairs=tokens**2,
                factorized_pairs=latent_frames * spatial**2 + spatial * latent_frames**2)


def flow_pair(clean, noise, tau):
    _paired(clean, noise)
    if not 0 <= tau <= 1:
        raise ValueError('tau must be in [0,1]')
    return ([(1-tau)*x + tau*e for x,e in zip(clean,noise)],
            [e-x for x,e in zip(clean,noise)])


def masked_mse(prediction, target, valid):
    """valid has one binary entry per scalar, after channel broadcasting."""
    _paired(prediction, target)
    _paired(prediction, valid)
    if any(v not in (0,1) for v in valid) or not sum(valid):
        raise ValueError('valid mask must be binary and contain a valid element')
    return sum((p-y)**2 for p,y,m in zip(prediction,target,valid) if m) / sum(valid)


def cfg(unconditional, conditional, scale):
    _paired(unconditional, conditional)
    return [u + scale*(c-u) for u,c in zip(unconditional,conditional)]


def euler(initial, times, velocity):
    """Return all states on a strictly decreasing noise-time grid."""
    if (not initial or len(times) < 2 or
        any(not 0 <= t <= 1 for t in times) or
        any(b >= a for a,b in zip(times,times[1:]))):
        raise ValueError('expected a nonempty state and descending times within [0,1]')
    state = list(initial)
    states = [state]
    for tau, following in zip(times,times[1:]):
        direction = velocity(state, tau)
        _paired(state, direction)
        state = [z + (following-tau)*v for z,v in zip(state,direction)]
        states.append(state)
    return states


def gaussian_velocity(state, tau, mean=2., std=.5):
    """Optimal marginal velocity for independent X~N(mean,std^2), E~N(0,1).

    Z_tau=(1-tau)X+tau E. Its mean is (1-tau)mean and variance is
    (1-tau)^2 std^2+tau^2; E[E-X | Z_tau=z] gives the expression below.
    This field is known analytically, not fitted by a neural network.
    """
    if std <= 0 or not math.isfinite(std) or not 0 <= tau <= 1:
        raise ValueError('expected positive finite std and tau within [0,1]')
    variance = (1-tau)**2 * std**2 + tau**2
    covariance = tau - (1-tau)*std**2
    return [-mean + covariance/variance * (z-(1-tau)*mean) for z in state]


def main():
    state, target = flow_pair([2.,-1.], [-2.,3.], .75)
    trajectory = euler([-2.,3.], [1.,.75,.5,.25,0.], lambda z,t: [-4.,4.])
    gaussian = euler([-1.,0.,1.], [1-i/2000 for i in range(2001)], gaussian_velocity)[-1]
    print(json.dumps(dict(budget=video_budget(), noisy_state=state, velocity_target=target,
                         masked_loss=masked_mse([-3.,5.,99.], [-4.,4.,0.], [1,1,0]),
                         constant_velocity_trajectory=trajectory,
                         gaussian_samples=[round(x,6) for x in gaussian],
                         gaussian_exact=[1.5,2.,2.5]), indent=2))


if __name__ == '__main__':
    main()
