"""TEST NHAN QUA: ban sua lech MOT BUOC co that khong?

`observe_sequence` truoc day dung `actions[:, t-1]` trong khi `obs[t]`
do `actions[t]` sinh ra. Ban sua doi kha nang dung hon, nhung:

  - doc code khong phai bang chung.
  - Do KL cua ca hai cach tren checkpoint da huanc o cau hinh CU cho ra
    0.4913 (dung) va 0.4879 (sai) — gan như bang nhau. Do la do tren
    mo hinh DA THICH NGHI voi quy uoc sai, nen no khong phan biet duoc.

Test nay do tren mo hinh HOC DUNG cach moi:
    Dung hanh dong thật   -> prior phai do khop tot
    Xao tron hanh dong     -> prior phai do te hon

Neu hai so nay cach nhau nhieu, prior THAT SU dung hanh dong, va ban
sua lech-mot-buoc da co hieu luc.

Khong dung lai the nao cung mot kiem chung: neu hai so bang nhau,
test bao "khong phan biet duoc" chu khong ket luan sai.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .soul import Soul
from .trainer import _kl_normal

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '_causal.ckpt')


def _banner(s):
    print(f'\n{"=" * 72}\n{s}\n{"=" * 72}')


def build(seed=0, steps=1200, warmup_episodes=30,
          learn_per_collect=20):
    cfg = Config()
    cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
    cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
    cfg.train.batch_size, cfg.train.seq_len = 8, 16
    torch.manual_seed(seed)
    s = Soul(cfg)
    rng = np.random.default_rng(seed)
    from .demo import PendulumEnv
    from .verify import collect_episodes
    env = PendulumEnv(seed=seed)
    collect_episodes(s, warmup_episodes, env, rng)
    while s.trainer.updates < steps:
        collect_episodes(s, 1, env, rng)
        for _ in range(learn_per_collect):
            s.learn(force=True)
    s.save(CKPT)
    return s


def _shuffle_action_timesteps(actions, valid, generator=None):
    """Shuffle time positions independently per sequence, preserving padding."""
    if actions.ndim != 3 or valid.shape != actions.shape[:2]:
        raise ValueError('actions must be [batch, time, action] and valid [batch, time]')
    shuffled = actions.clone()
    for batch_index in range(actions.shape[0]):
        indices = torch.nonzero(valid[batch_index] > 0, as_tuple=False).flatten()
        if indices.numel() > 1:
            permutation = torch.randperm(
                indices.numel(), device=actions.device, generator=generator
            )
            shuffled[batch_index, indices] = actions[batch_index, indices[permutation]]
    return shuffled


INTERVENTIONS = ('real', 'zero', 'shuffle', 'lag', 'lead', 'negate')


def intervene_actions(actions, valid, kind='real', generator=None):
    """Apply one intervention; only valid timesteps may change.

    Lag/lead use an explicit zero at the sequence boundary instead of
    borrowing an action from the other side of an episode.
    """
    if kind not in INTERVENTIONS:
        raise ValueError(f'unknown action intervention: {kind!r}')
    if actions.ndim != 3 or valid.shape != actions.shape[:2]:
        raise ValueError('actions must be [batch, time, action] and valid [batch, time]')
    if kind == 'real':
        return actions.clone()

    mask = valid > 0
    result = actions.clone()
    if kind == 'zero':
        result[mask] = 0
        return result
    if kind == 'negate':
        result[mask] = -result[mask]
        return result

    for batch_index in range(actions.shape[0]):
        indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        values = actions[batch_index, indices].clone()
        if kind == 'shuffle':
            one = actions[batch_index:batch_index + 1]
            one_valid = valid[batch_index:batch_index + 1]
            replaced = _shuffle_action_timesteps(one, one_valid, generator=generator)[0, indices]
        else:
            replaced = torch.zeros_like(values)
            if indices.numel() > 1:
                if kind == 'lag':
                    replaced[1:] = values[:-1]
                else:
                    replaced[:-1] = values[1:]
        result[batch_index, indices] = replaced
    return result


def measure_kl_series(s, rng, intervention='real', n_batches=10,
                      intervention_seed=0):
    """Return per-batch KL values and the number of valid transitions.

    Intervention permutations use a private generator so they never consume
    the latent-noise stream shared by every paired measurement.
    """
    if intervention not in INTERVENTIONS:
        raise ValueError(f'unknown action intervention: {intervention!r}')
    values = []
    valid_transitions = 0
    generator = None
    for _ in range(n_batches):
        b = s.replay.sample(8, rng)
        if b is None:
            continue
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
        ac, va = t(b['actions']), t(b['valid'])
        seq = ac.shape[1]
        oi = torch.cat([v_, m_, tx], -1)[:, :seq]
        if intervention == 'shuffle' and generator is None:
            generator = torch.Generator(device=ac.device)
            generator.manual_seed(intervention_seed)
        ac = intervene_actions(ac, va, intervention, generator=generator)
        with torch.no_grad():
            out = s.model.dynamics.observe_sequence(
                oi, ac, s.model.encoder, va)
            kl = _kl_normal(out['prior_mu'], out['prior_lv'],
                            out['post_mu'], out['post_lv']).sum(-1)
            kl = torch.maximum(kl, torch.tensor(0.0)).mean()
        values.append(float(kl))
        valid_transitions += int((va > 0).sum().item())
    return values, valid_transitions


def measure_ablation(s, seed, n_batches=10, interventions=INTERVENTIONS):
    """Measure every intervention from one paired latent-noise stream."""
    kinds = tuple(interventions)
    if not kinds or len(set(kinds)) != len(kinds):
        raise ValueError('interventions must be a non-empty list of unique names')
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if cuda_devices else None
        series = {}
        valid_transitions = 0
        for offset, kind in enumerate(kinds):
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            values, count = measure_kl_series(
                s, np.random.default_rng(seed), kind,
                n_batches=n_batches, intervention_seed=seed + 1000 + offset,
            )
            series[kind] = values
            valid_transitions = count
    return {'series': series, 'valid_transitions': valid_transitions}


def _stats(values):
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'dispersion': float(np.std(values)),
        'count': len(values),
    }


def summarize_ablation(result):
    """Per-intervention KL plus paired effects against the real-action run."""
    series = result['series']
    real = series.get('real')
    interventions = {
        kind: dict(_stats(values), per_episode_kl=list(values))
        for kind, values in series.items()
    }
    paired = {}
    if real is not None:
        paired = {
            kind: _stats([value - baseline for value, baseline in zip(values, real)])
            for kind, values in series.items() if kind != 'real'
        }
    return {
        'interventions': interventions,
        'paired_effects_vs_real': paired,
        'valid_transitions': result['valid_transitions'],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def measure(s, rng, shuffle, n_batches=10, shuffle_seed=0):
    values, _ = measure_kl_series(
        s, rng, 'shuffle' if shuffle else 'real',
        n_batches=n_batches, intervention_seed=shuffle_seed,
    )
    return sum(values) / max(len(values), 1)


def measure_paired_actions(s, seed, n_batches=10):
    """Compare paired interventions without changing the caller's global RNG state."""
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    # The context restores the caller's CPU and CUDA RNG states even if either
    # measurement raises. Inside it, both interventions still start from the
    # same seeded latent-noise stream.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if cuda_devices else None

        real = measure(
            s,
            np.random.default_rng(seed),
            False,
            n_batches=n_batches,
            shuffle_seed=seed + 1,
        )
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        shuffled = measure(
            s,
            np.random.default_rng(seed),
            True,
            n_batches=n_batches,
            shuffle_seed=seed + 1,
        )
    return real, shuffled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--warmup-episodes', type=int, default=30)
    ap.add_argument('--learn-per-collect', type=int, default=20)
    ap.add_argument('--batches', type=int, default=10)
    ap.add_argument('--matrix', action='store_true',
                    help='measure all action interventions with paired RNG')
    ap.add_argument('--report', help='write a machine-readable JSON report')
    ap.add_argument('--no-train', action='store_true',
                    help='refuse to train when the diagnostic checkpoint is absent')
    ap.add_argument('--train-path', help='optional TRAIN snapshot to hash into the report')
    ap.add_argument('--pilot-path', help='optional PILOT snapshot to hash into the report')
    args = ap.parse_args()
    if args.report and not args.matrix:
        ap.error('--report requires --matrix')
    if not os.path.exists(CKPT) and args.no_train:
        raise SystemExit(f'checkpoint is missing and training is disabled: {CKPT}')
    if os.path.exists(CKPT):
        print(f'  Load checkpoint: {CKPT}')
        cfg = Config()
        cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
        cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
        cfg.train.batch_size, cfg.train.seq_len = 8, 16
        s = Soul(cfg)
        s.load(CKPT)
    else:
        _banner('Huan 1200 buoc (lan dau)')
        s = build(args.seed, args.steps, args.warmup_episodes,
                  args.learn_per_collect)
        print('  xong.')

    if args.matrix:
        result = measure_ablation(s, args.seed + 7, n_batches=args.batches)
        summary = summarize_ablation(result)
        _banner('Action intervention matrix (exploratory; DRAFT_NOT_LOCKED)')
        for kind, values in result['series'].items():
            stats = summary['interventions'][kind]
            print(f'  {kind:8s}: KL mean={stats["mean"]:.4f} '
                  f'dispersion={stats["dispersion"]:.4f} batches={stats["count"]}')
        for kind, effect in summary['paired_effects_vs_real'].items():
            print(f'  {kind:8s} minus real: {effect["mean"]:+.4f} '
                  f'(median {effect["median"]:+.4f}, dispersion {effect["dispersion"]:.4f})')
        print(f'  valid transitions per measurement: {summary["valid_transitions"]}')
        print('  No threshold or promotion decision is made by this diagnostic.')
        if args.report:
            report_path = Path(args.report)
            if report_path.resolve() == Path(CKPT).resolve():
                raise SystemExit('report path must not overwrite the checkpoint')
            payload = {
                'protocol': 'DRAFT_NOT_LOCKED',
                'threshold': 'TO_LOCK',
                'confirm_accessed': False,
                'interpretation': 'exploratory only; no generalization or promotion claim',
                'provenance': {
                    'checkpoint_path': str(CKPT),
                    'checkpoint_sha256': sha256_file(CKPT),
                    'train_sha256': sha256_file(args.train_path) if args.train_path else None,
                    'pilot_sha256': sha256_file(args.pilot_path) if args.pilot_path else None,
                    'code_commit': os.environ.get('GITHUB_SHA'),
                    'python': sys.version.split()[0],
                    'torch': torch.__version__,
                    'numpy': np.__version__,
                    'device': str(getattr(s, 'device', 'unknown')),
                    'seed': args.seed,
                    'n_batches': args.batches,
                    'interventions': list(INTERVENTIONS),
                },
                'summary': summary,
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open('w', encoding='utf-8') as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write('\n')
            print(f'  report: {report_path}')
        return

    _banner('KL prior||posterior cua hanh dung va hanh xao tron')
    real, shuf = measure_paired_actions(s, args.seed + 7, n_batches=args.batches)
    print(f'  hanh dong THAT     : KL = {real:.4f} nats')
    print(f'  hanh dong XAO TRON : KL = {shuf:.4f} nats')
    print(f'  chenh lech         : {shuf - real:+.4f} nats')

    _banner('Ket luan')
    if shuf > real * 1.05:
        print('  Prior DO TOT HON khi dung hanh dong that. -> prior that su')
        print('  dung hanh dong, va ban sua lech-mot-buoc CO hieu luc.')
    elif shuf < real:
        print('  KHONG phan biet duoc (xao tron ra KL nho hon). Ban sua')
        print('  lech-mot-buoc co the KHONG co tac dung — can xem lai.')
    else:
        print('  KHONG phan biet duoc: hai so gan bang nhau. Chua co bang')
        print('  chung nao rang prior co that dung hanh dong hay khong.')


if __name__ == '__main__':
    sys.exit(main())
