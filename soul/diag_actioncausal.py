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
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .soul import Soul
from .trainer import _kl_normal

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '_causal.ckpt')
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_PROVENANCE_FILE_BYTES = 256 * 1024 * 1024


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


def _masked_kl_mean(kl, valid):
    """Average KL over real transitions, excluding terminal/padded boundary rows."""
    if kl.ndim != 2 or valid.shape != kl.shape:
        raise ValueError('KL and valid-transition mask must have matching [batch, time] shapes')
    if not torch.isfinite(valid).all() or not torch.logical_or(valid == 0, valid == 1).all():
        raise ValueError('valid-transition mask must contain only finite zeros and ones')
    selected = kl.masked_select(valid > 0)
    if selected.numel() == 0:
        raise ValueError('KL batch contains no valid transitions')
    if not torch.isfinite(selected).all():
        raise ValueError('valid-transition KL values must be finite')
    return selected.clamp_min(0.0).mean()


def measure_kl_series(s, rng, intervention='real', n_batches=10,
                      intervention_seed=0):
    """Return per-batch valid-transition KL values and the transition count.

    Terminal/padded boundary rows are excluded from the mean. Intervention
    permutations use a private generator so they never consume the latent-noise
    stream shared by every paired measurement.
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
            kl = _masked_kl_mean(kl, va)
        values.append(float(kl))
        valid_transitions += int((va > 0).sum().item())
    return values, valid_transitions


def measure_ablation(s, seed, n_batches=10, interventions=INTERVENTIONS):
    """Measure every intervention from one paired latent-noise stream."""
    if isinstance(n_batches, bool) or not isinstance(n_batches, int) or n_batches <= 0:
        raise ValueError('n_batches must be a positive integer')
    kinds = tuple(interventions)
    if not kinds or len(set(kinds)) != len(kinds):
        raise ValueError('interventions must be a non-empty list of unique names')
    for kind in kinds:
        if kind not in INTERVENTIONS:
            raise ValueError(f'unknown action intervention: {kind!r}')
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
        valid_transitions = None
        for offset, kind in enumerate(kinds):
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            values, count = measure_kl_series(
                s, np.random.default_rng(seed), kind,
                n_batches=n_batches, intervention_seed=seed + 1000 + offset,
            )
            if len(values) != n_batches:
                raise ValueError(
                    f'{kind} intervention returned {len(values)} batches; '
                    f'expected {n_batches}'
                )
            if valid_transitions is None:
                valid_transitions = count
            elif count != valid_transitions:
                raise ValueError(
                    f'{kind} intervention valid-transition count differs from '
                    f'the paired baseline ({count} != {valid_transitions})'
                )
            series[kind] = values
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
    if (
        not isinstance(result, dict)
        or not isinstance(result.get('series'), dict)
        or not result['series']
    ):
        raise ValueError('ablation result must contain non-empty intervention series')

    series = result['series']
    lengths = set()
    for kind, values in series.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f'{kind} intervention series must contain at least one batch')
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            for value in values
        ):
            raise ValueError(f'{kind} intervention series must contain finite numeric values')
        lengths.add(len(values))
    if len(lengths) != 1:
        raise ValueError('all intervention series must have the same number of paired batches')
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
    """Hash a bounded regular provenance file without following links or blocking on special files."""
    path = Path(path)
    try:
        path_before = path.lstat()
    except OSError as error:
        raise ValueError(f'provenance input is unavailable: {path}') from error
    if (
        _checkpoint_is_link_or_reparse_point(path_before)
        or not stat.S_ISREG(path_before.st_mode)
    ):
        raise ValueError(f'provenance input must be a regular non-symlink file: {path}')

    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f'provenance input could not be opened safely: {path}: {error}') from error

    digest = hashlib.sha256()
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if (
            _checkpoint_is_link_or_reparse_point(before)
            or not stat.S_ISREG(before.st_mode)
            or _checkpoint_file_identity(before) != _checkpoint_file_identity(path_before)
        ):
            raise ValueError(f'provenance input changed while it was being opened: {path}')
        try:
            path_after_open = path.lstat()
        except OSError as error:
            raise ValueError(f'provenance input path changed while it was being opened: {path}') from error
        if (
            _checkpoint_is_link_or_reparse_point(path_after_open)
            or not stat.S_ISREG(path_after_open.st_mode)
            or _checkpoint_file_identity(path_after_open) != _checkpoint_file_identity(before)
        ):
            raise ValueError(f'provenance input path changed while it was being opened: {path}')
        if before.st_size > MAX_PROVENANCE_FILE_BYTES:
            raise ValueError(
                f'provenance input exceeds size limit: {path} '
                f'(limit {MAX_PROVENANCE_FILE_BYTES} bytes)'
            )

        total = 0
        while True:
            chunk = stream.read(min(1024 * 1024, MAX_PROVENANCE_FILE_BYTES - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_PROVENANCE_FILE_BYTES:
                raise ValueError(
                    f'provenance input exceeds size limit: {path} '
                    f'(limit {MAX_PROVENANCE_FILE_BYTES} bytes)'
                )
            digest.update(chunk)

        after = os.fstat(stream.fileno())
        try:
            path_after = path.lstat()
        except OSError as error:
            raise ValueError(f'provenance input path changed while it was being hashed: {path}') from error
        if (
            total != before.st_size
            or _checkpoint_file_identity(before) != _checkpoint_file_identity(after)
            or _checkpoint_file_identity(after) != _checkpoint_file_identity(path_after)
            or _checkpoint_is_link_or_reparse_point(path_after)
            or not stat.S_ISREG(path_after.st_mode)
        ):
            raise ValueError(f'provenance input changed while it was being hashed: {path}')
    return digest.hexdigest()


def _checkpoint_file_identity(metadata):
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
    if os.name != 'nt':
        return (*identity, metadata.st_ctime_ns)
    return identity


def _checkpoint_is_link_or_reparse_point(metadata):
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, 'st_file_attributes', 0) & 0x400
    )


def checkpoint_exists_or_training(path, allow_training=False):
    """Fail closed unless an existing checkpoint is present or training is explicit."""
    checkpoint = Path(path)
    if checkpoint.is_symlink():
        raise ValueError(f'diagnostic checkpoint must not be a symlink: {checkpoint}')
    if checkpoint.is_file():
        return True
    if allow_training:
        return False
    raise FileNotFoundError(
        f'frozen diagnostic checkpoint is missing: {checkpoint}; '
        'pass --train-if-missing only for an explicit training run'
    )


def load_checkpoint_snapshot(soul, checkpoint_path, expected_sha256=None):
    """Hash and load one bounded, no-follow disk snapshot of the checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    expected = None
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str):
            raise ValueError('expected checkpoint SHA-256 must be a string')
        expected = expected_sha256.strip().lower()
        if len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
            raise ValueError('expected checkpoint SHA-256 must be 64 lowercase hexadecimal characters')

    try:
        path_before = checkpoint_path.lstat()
    except OSError as error:
        raise ValueError(f'diagnostic checkpoint is unavailable: {checkpoint_path}') from error
    if (
        _checkpoint_is_link_or_reparse_point(path_before)
        or not stat.S_ISREG(path_before.st_mode)
    ):
        raise ValueError(
            f'diagnostic checkpoint must be a regular non-symlink file: {checkpoint_path}'
        )

    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(checkpoint_path, flags)
    except OSError as error:
        raise ValueError(f'diagnostic checkpoint could not be opened safely: {checkpoint_path}: {error}') from error

    digest = hashlib.sha256()
    snapshot_path = None
    try:
        with os.fdopen(descriptor, 'rb') as source:
            before = os.fstat(source.fileno())
            if (
                _checkpoint_is_link_or_reparse_point(before)
                or not stat.S_ISREG(before.st_mode)
                or _checkpoint_file_identity(before) != _checkpoint_file_identity(path_before)
            ):
                raise ValueError('diagnostic checkpoint path changed while it was being opened')
            try:
                path_after_open = checkpoint_path.lstat()
            except OSError as error:
                raise ValueError('diagnostic checkpoint path changed while it was being opened') from error
            if (
                _checkpoint_is_link_or_reparse_point(path_after_open)
                or not stat.S_ISREG(path_after_open.st_mode)
                or _checkpoint_file_identity(path_after_open) != _checkpoint_file_identity(before)
            ):
                raise ValueError('diagnostic checkpoint path changed while it was being opened')
            if before.st_size <= 0:
                raise ValueError('diagnostic checkpoint is empty')
            if before.st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError(
                    f'diagnostic checkpoint exceeds size limit '
                    f'({before.st_size} bytes; limit {MAX_CHECKPOINT_BYTES})'
                )

            with tempfile.NamedTemporaryFile(
                mode='wb',
                prefix='soul-action-',
                suffix='.ckpt',
                delete=False,
            ) as snapshot:
                snapshot_path = Path(snapshot.name)
                total = 0
                while True:
                    chunk = source.read(min(1024 * 1024, MAX_CHECKPOINT_BYTES - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_CHECKPOINT_BYTES:
                        raise ValueError(
                            f'diagnostic checkpoint exceeds size limit '
                            f'(limit {MAX_CHECKPOINT_BYTES} bytes)'
                        )
                    digest.update(chunk)
                    snapshot.write(chunk)
                if total != before.st_size:
                    raise ValueError('diagnostic checkpoint size changed while creating its snapshot')
                snapshot.flush()
                os.fsync(snapshot.fileno())

            after = os.fstat(source.fileno())
            try:
                path_after = checkpoint_path.lstat()
            except OSError as error:
                raise ValueError('diagnostic checkpoint path changed during snapshot creation') from error
            if (
                _checkpoint_is_link_or_reparse_point(path_after)
                or not stat.S_ISREG(path_after.st_mode)
                or _checkpoint_file_identity(before) != _checkpoint_file_identity(after)
                or _checkpoint_file_identity(after) != _checkpoint_file_identity(path_after)
            ):
                raise ValueError('diagnostic checkpoint changed during snapshot creation')

        actual_sha256 = digest.hexdigest()
        if expected is not None and actual_sha256 != expected:
            raise ValueError('diagnostic checkpoint SHA-256 does not match the expected digest')

        soul.load(str(snapshot_path))
        return actual_sha256
    finally:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)


def _reject_report_output_aliases(report_path, protected_paths):
    """Refuse report targets that alias the checkpoint or hashed source datasets."""
    report_path = Path(report_path)
    if report_path.is_symlink():
        raise ValueError('report path must not be a symlink')
    if report_path.exists() and not report_path.is_file():
        raise ValueError('report path must be a regular file')

    try:
        resolved_report = report_path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f'cannot resolve report path: {report_path}') from error

    for protected in protected_paths:
        if protected is None:
            continue
        source = Path(protected)
        try:
            if resolved_report == source.resolve(strict=False):
                raise ValueError(
                    f'report path must not overwrite protected input: {source}'
                )
            if report_path.exists() and source.exists() and os.path.samefile(report_path, source):
                raise ValueError(
                    f'report path must not overwrite protected input: {source}'
                )
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f'cannot verify report path against protected input: {source}'
            ) from error


def _write_json_report_atomic(report_path, payload, protected_paths):
    """Write a report beside its destination and replace it only after a complete flush."""
    report_path = Path(report_path)
    _reject_report_output_aliases(report_path, protected_paths)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_report_output_aliases(report_path, protected_paths)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=report_path.parent,
            prefix=f'.{report_path.name}.',
            suffix='.tmp',
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())

        # Re-check immediately before replacement. os.replace never follows an
        # existing destination symlink, but refusing it keeps the path contract clear.
        _reject_report_output_aliases(report_path, protected_paths)
        os.replace(temporary_path, report_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


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
    training = ap.add_mutually_exclusive_group()
    training.add_argument(
        '--train-if-missing',
        dest='allow_training',
        action='store_true',
        help='explicitly allow a new training run when the diagnostic checkpoint is absent',
    )
    training.add_argument(
        '--no-train',
        dest='allow_training',
        action='store_false',
        help='refuse training when the diagnostic checkpoint is absent (default)',
    )
    ap.set_defaults(allow_training=False)
    ap.add_argument('--expected-checkpoint-sha256', help='optional digest required before checkpoint deserialization')
    ap.add_argument(
        '--train-path',
        help='optional TRAIN path hashed for provenance only; not used as scoring input',
    )
    ap.add_argument(
        '--pilot-path',
        help='optional PILOT path hashed for provenance only; not used as scoring input',
    )
    args = ap.parse_args()
    if args.report and not args.matrix:
        ap.error('--report requires --matrix')
    protected_report_inputs = [CKPT, args.train_path, args.pilot_path]
    if args.report:
        try:
            _reject_report_output_aliases(Path(args.report), protected_report_inputs)
        except ValueError as error:
            ap.error(str(error))
    try:
        checkpoint_exists = checkpoint_exists_or_training(CKPT, args.allow_training)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not checkpoint_exists and args.expected_checkpoint_sha256:
        raise SystemExit('--expected-checkpoint-sha256 requires an existing checkpoint')

    checkpoint_sha256 = None
    if checkpoint_exists:
        print(f'  Load checkpoint snapshot: {CKPT}')
        cfg = Config()
        cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
        cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
        cfg.train.batch_size, cfg.train.seq_len = 8, 16
        s = Soul(cfg)
        checkpoint_sha256 = load_checkpoint_snapshot(
            s, CKPT, expected_sha256=args.expected_checkpoint_sha256
        )
    else:
        _banner('Huan 1200 buoc (explicit training opt-in)')
        s = build(args.seed, args.steps, args.warmup_episodes,
                  args.learn_per_collect)
        checkpoint_sha256 = sha256_file(CKPT)
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
            payload = {
                'protocol': 'DRAFT_NOT_LOCKED',
                'threshold': 'TO_LOCK',
                'confirm_accessed': False,
                'interpretation': 'exploratory only; no generalization or promotion claim',
                'provenance': {
                    'checkpoint_path': str(CKPT),
                    'checkpoint_sha256': checkpoint_sha256,
                    'checkpoint_loaded_from_existing_file': checkpoint_exists,
                    'replay_source': 'checkpoint_embedded_replay_buffer',
                    'external_dataset_paths_used_for_scoring': False,
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
                'limitations': [
                    'The matrix samples the replay buffer embedded in the checkpoint.',
                    'Optional --train-path and --pilot-path values are hashed only; they are not scoring inputs.',
                    'Physical prediction deltas are not measured by this diagnostic.',
                ],
            }
            _write_json_report_atomic(
                report_path, payload, protected_report_inputs
            )
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
