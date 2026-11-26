import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from soul.config import Config, ModelConfig
from soul.demo import PendulumEnv
from soul.obs import ObsEncoder
from soul.soul import Soul


class SoulRegressionTests(unittest.TestCase):
    def test_plan_elite_must_be_positive(self):
        # topk(..., 0) yields an empty elite set and silently poisons CEM with NaN.
        with self.assertRaisesRegex(ValueError, 'plan_elite'):
            ModelConfig(plan_elite=0)

    def test_observation_budgets_must_be_positive(self):
        from soul.config import ObsConfig

        with self.assertRaisesRegex(ValueError, 'max_features'):
            ObsConfig(max_features=0)
        with self.assertRaisesRegex(ValueError, 'text_buckets'):
            ObsConfig(text_buckets=0)

    def test_action_columns_keep_their_own_dimension(self):
        from soul.dataset import _to_action

        action = _to_action({'a1': '0.75'}, action_dim=3)
        np.testing.assert_allclose(
            action, np.array([0.0, 0.75, 0.0], dtype=np.float32))

    def test_time_feature_keeps_information_across_long_horizon(self):
        enc = ObsEncoder()
        t0 = enc.encode({'time': 0})['values'][enc._schema['time']]
        t50 = enc.encode({'time': 50})['values'][enc._schema['time']]
        t200 = enc.encode({'time': 200})['values'][enc._schema['time']]

        self.assertGreater(t50 - t0, 0.20)
        self.assertGreater(t200 - t50, 0.20)

    def test_pendulum_env_clips_action_to_safe_range(self):
        env_clipped = PendulumEnv(seed=7)
        env_raw = PendulumEnv(seed=7)
        env_clipped.reset()
        env_raw.reset()

        _, _, _, _ = env_clipped.step(np.array([3.0], dtype=np.float32))
        _, _, _, _ = env_raw.step(np.array([1.0], dtype=np.float32))

        self.assertAlmostEqual(env_clipped.x, env_raw.x, places=5)
        self.assertAlmostEqual(env_clipped.v, env_raw.v, places=5)

    def test_demo_records_applied_action(self):
        from soul.demo import run_episode

        class FakeEnv:
            def reset(self, rng=None):
                return self._obs()

            def _obs(self):
                return {'state': {'position': 0.0, 'velocity': 0.0}, 'time': 0}

            def step(self, action):
                applied = np.clip(action, -1.0, 1.0)
                return self._obs(), 0.0, True, {'applied_action': applied}

            def goal_vec(self):
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        class RecordingSoul:
            def __init__(self):
                self.actions = []

            def set_goal(self, goal):
                pass

            def act(self, obs, deterministic=False):
                return np.array([2.0], dtype=np.float32)

            def observe(self, obs, action, reward, done, applied_action=None):
                self.actions.append(
                    np.asarray(action if applied_action is None else applied_action))

        soul = RecordingSoul()
        run_episode(soul, FakeEnv(), None, train=False)
        np.testing.assert_allclose(soul.actions[0], [1.0])

    def test_reward_stats_stay_finite_for_one_row_batch(self):
        cfg = Config()
        cfg.obs.max_features = 4
        cfg.obs.text_buckets = 4
        cfg.model.hidden_dim = 8
        cfg.model.latent_dim = 4
        cfg.model.action_dim = 1
        cfg.model.goal_dim = 2
        cfg.train.imagine_horizon = 2
        cfg.train.seq_len = 1
        soul = Soul(cfg)

        batch = {
            'values': np.zeros((1, 2, cfg.obs.max_features), np.float32),
            'masks': np.ones((1, 2, cfg.obs.max_features), np.float32),
            'texts': np.zeros((1, 2, cfg.obs.text_buckets), np.float32),
            'actions': np.zeros((1, 1, cfg.model.action_dim), np.float32),
            'rewards': np.full((1, 1), -0.5, np.float32),
            'continues': np.ones((1, 1), np.float32),
            'goals': np.zeros((1, 1, cfg.model.goal_dim), np.float32),
            'valid': np.ones((1, 1), np.float32),
        }

        losses = soul.trainer.update(batch)

        self.assertTrue(np.isfinite(losses['reward']))
        self.assertTrue(np.isfinite(soul.trainer._reward_stats['std']))

    def test_action_effect_predictor_is_available(self):
        cfg = Config()
        cfg.model.hidden_dim = 32
        cfg.model.latent_dim = 8
        cfg.train.batch_size = 2
        cfg.train.seq_len = 4
        soul = Soul(cfg)

        obs = {'time': 0.0, 'x': 0.0, 'y': 0.0}
        action = np.array([0.25], dtype=np.float32)
        pred = soul.predict_action_effect(obs, action)

        self.assertEqual(pred.shape, (cfg.model.latent_dim,))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_planned_action_uses_prediction(self):
        cfg = Config()
        cfg.model.hidden_dim = 32
        cfg.model.latent_dim = 8
        cfg.model.plan_samples = 8
        cfg.train.batch_size = 2
        cfg.train.seq_len = 4
        soul = Soul(cfg)

        obs = {'time': 0.0, 'x': 0.0, 'y': 0.0}
        action = soul.plan_action(obs, n_samples=8)

        self.assertEqual(action.shape, (cfg.model.action_dim,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(np.abs(action) <= 4.0))

    def test_brain_has_cognition_modules(self):
        cfg = Config()
        cfg.model.hidden_dim = 32
        cfg.model.latent_dim = 8
        cfg.train.batch_size = 2
        cfg.train.seq_len = 4
        soul = Soul(cfg)

        obs = {'time': 0.0, 'x': 0.0, 'y': 0.0}
        brain = soul.brain

        state = brain.perceive(obs)
        pred = brain.predict(obs, np.array([0.25], dtype=np.float32))
        action = brain.plan(obs, n_samples=4)

        self.assertEqual(state.shape[0], soul.enc.input_dim)
        self.assertEqual(pred.shape, (cfg.model.latent_dim,))
        self.assertEqual(action.shape, (cfg.model.action_dim,))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_skill_memory_extracts_and_recalls(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        event = {
            'time': 12,
            'reward': 0.8,
            'done': False,
            'action': np.array([0.5], dtype=np.float32),
            'context': {'goal': 'forward', 'risk': 'low'},
        }

        skill = mgr.extract_skill_from_event(event)
        self.assertIsNotNone(skill)
        self.assertTrue(skill['name'])
        self.assertGreater(len(mgr.skills.best()), 0)
        self.assertGreater(len(mgr.recall_skills({'goal': 'forward'})), 0)

    def test_archive_memory_can_persist_to_disk(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        event = {
            'time': 21,
            'reward': -0.3,
            'done': False,
            'action': np.array([0.2], dtype=np.float32),
            'context': {'goal': 'forward', 'risk': 'medium'},
        }
        mgr.record_event(event, important=True)
        mgr.extract_skill_from_event(event)

        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            mgr.archive.save(path)
            restored = MemoryManager()
            restored.archive.load(path)
            self.assertGreater(len(restored.archive.events), 0)
            self.assertGreater(len(restored.archive.skills), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_memory_load_respects_capacity(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        mgr.archive.max_events = 2
        mgr.archive.max_skills = 2
        mgr.skills.maxlen = 2
        payload = {
            'archive': {
                'events': [{'time': i} for i in range(5)],
                'skills': [{'name': f'a{i}', 'confidence': 1.0}
                           for i in range(5)],
            },
            'skills': [{'name': f's{i}', 'confidence': 1.0}
                       for i in range(5)],
        }

        mgr.from_dict(payload)

        self.assertEqual([e['time'] for e in mgr.archive.events], [3, 4])
        self.assertEqual([s['name'] for s in mgr.archive.skills], ['a3', 'a4'])
        self.assertEqual([s['name'] for s in mgr.skills.skills], ['s3', 's4'])

    def test_skill_confidence_decays_over_time(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        event = {
            'time': 5,
            'reward': 0.9,
            'done': False,
            'action': np.array([0.6], dtype=np.float32),
            'context': {'goal': 'forward', 'risk': 'low'},
        }
        skill = mgr.extract_skill_from_event(event)
        self.assertGreater(skill['confidence'], 0.5)

        mgr.skills.skills[0]['last_seen'] = 0
        mgr.refresh_skills(current_time=50, decay=0.08)
        self.assertLess(mgr.skills.skills[0]['confidence'], 0.9)

    def test_archive_summary_creates_reusable_skill(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        for i in range(3):
            event = {
                'time': i,
                'reward': 0.7 + 0.1 * i,
                'done': False,
                'action': np.array([0.5 + 0.2 * i], dtype=np.float32),
                'context': {'goal': 'forward', 'risk': 'low'},
            }
            mgr.record_event(event, important=True)
            mgr.extract_skill_from_event(event)

        skills = mgr.summarize_archive_to_skills(min_events=2, max_skills=4)
        self.assertTrue(skills)
        self.assertTrue(any(skill.get('context', {}).get('goal') == 'forward' for skill in skills))

    def test_soul_archive_can_persist_and_restore(self):
        cfg = Config()
        soul = Soul(cfg)
        event = {
            'time': 7,
            'reward': 0.8,
            'done': False,
            'action': np.array([0.25], dtype=np.float32),
            'context': {'goal': 'forward'},
        }
        soul.memory_manager.record_event(event, important=True)
        soul.memory_manager.extract_skill_from_event(event)

        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            soul.save_archive(path)
            restored = Soul(cfg)
            restored.load_archive(path)
            self.assertGreater(len(restored.memory_manager.archive.events), 0)
            self.assertGreater(len(restored.memory_manager.skills.skills), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)


    def test_checkpoint_cannot_select_an_archive_to_load(self):
        import torch

        cfg = Config()
        fd, path = tempfile.mkstemp(suffix='.pt')
        os.close(fd)
        try:
            Soul(cfg).save(path)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            ckpt['archive_path'] = path
            torch.save(ckpt, path)
            # The embedded 'memory' is authoritative; a checkpoint must not
            # name an arbitrary JSON file to load on top of it.
            with patch.object(
                Soul, 'load_archive',
                side_effect=AssertionError('checkpoint-controlled archive path followed'),
            ):
                restored = Soul(cfg)
                restored.load(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_read_npz_rejects_wrong_action_width(self):
        from soul.dataset import read_npz

        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            np.savez(path, action=np.zeros((2, 2), np.float32),
                     reward=np.zeros(2, np.float32))
            with self.assertRaisesRegex(ValueError, 'action table'):
                read_npz(path, action_dim=3, goal_dim=1)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_read_npz_rejects_short_observation_field(self):
        from soul.dataset import read_npz

        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            np.savez(path, action=np.zeros((2, 3), np.float32),
                     obs__x=np.zeros(1, np.float32))
            with self.assertRaisesRegex(ValueError, 'obs__x'):
                read_npz(path, action_dim=3, goal_dim=1)
        finally:
            if os.path.exists(path):
                os.remove(path)
    def test_train_resume_requires_allowlisted_digest(self):
        from soul import train

        with patch('sys.argv', ['train', '--resume', 'untrusted.pt']):
            with self.assertRaisesRegex(SystemExit, '--resume requires --resume-sha256'):
                train.main()

    def test_plan_existing_checkpoint_requires_allowlisted_digest(self):
        from soul import plan

        fd, path = tempfile.mkstemp(suffix='.ckpt')
        os.close(fd)
        try:
            with patch('sys.argv', ['plan', '--ckpt', path]):
                with self.assertRaisesRegex(SystemExit, 'requires --ckpt-sha256'):
                    plan.main()
        finally:
            os.remove(path)

    def test_read_npz_accepts_empty_action_table(self):
        from soul.dataset import read_npz

        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            np.savez(path, action=np.zeros((0, 3), np.float32))
            self.assertEqual(read_npz(path, action_dim=3, goal_dim=1), [])
        finally:
            os.remove(path)

    def test_read_npz_rejects_multidimensional_observation_field(self):
        from soul.dataset import read_npz

        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            np.savez(path, action=np.zeros((2, 3), np.float32),
                     obs__x=np.zeros((2, 2), np.float32))
            with self.assertRaisesRegex(ValueError, 'one-dimensional'):
                read_npz(path, action_dim=3, goal_dim=1)
        finally:
            os.remove(path)

    def test_import_helpers_enforce_byte_and_row_caps(self):
        import soul.dataset as dataset

        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        try:
            with open(path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['ep', 'done', 'reward', 'a0', 'x'])
                for i in range(6):
                    writer.writerow([0, i == 5, 0.0, 0.1, float(i)])

            with patch.object(dataset, 'MAX_IMPORT_BYTES', 4):
                with patch('builtins.open', side_effect=AssertionError('must not open')):
                    with self.assertRaisesRegex(ValueError, 'byte limit'):
                        dataset.read_csv(path, action_dim=1, goal_dim=0)
            with patch.object(dataset, 'MAX_IMPORT_ROWS', 3):
                with self.assertRaisesRegex(ValueError, 'row limit'):
                    dataset.read_csv(path, action_dim=1, goal_dim=0)
        finally:
            os.remove(path)

        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            np.savez(path, action=np.zeros((6, 1), np.float32),
                     reward=np.zeros(6, np.float32),
                     done=np.zeros(6, np.int64), ep=np.zeros(6, np.int64))
            with patch.object(dataset, 'MAX_IMPORT_BYTES', 4):
                with patch.object(dataset.np, 'load',
                                  side_effect=AssertionError('must not load')):
                    with self.assertRaisesRegex(ValueError, 'byte limit'):
                        dataset.read_npz(path, action_dim=1, goal_dim=0)
            with patch.object(dataset, 'MAX_IMPORT_ROWS', 3):
                with self.assertRaisesRegex(ValueError, 'row limit'):
                    dataset.read_npz(path, action_dim=1, goal_dim=0)
        finally:
            os.remove(path)

    def test_csv_import_caps_bytes_even_when_size_is_misreported(self):
        import soul.dataset as dataset

        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        try:
            with open(path, 'w', newline='', encoding='utf-8') as handle:
                handle.write('ep,done,reward,a0,x' + chr(10))
                handle.write('0,0,0.0,0.1,' + 'y' * 200 + chr(10))
            with patch.object(dataset, 'MAX_IMPORT_BYTES', 32), patch.object(dataset.os.path, 'getsize', return_value=0):
                with self.assertRaisesRegex(ValueError, 'byte limit'):
                    dataset.read_csv(path, action_dim=1, goal_dim=0)
        finally:
            os.remove(path)

    def test_csv_import_refuses_linked_file(self):
        import soul.dataset as dataset

        with tempfile.TemporaryDirectory() as dirname:
            victim = os.path.join(dirname, 'victim.csv')
            with open(victim, 'w', newline='', encoding='utf-8') as handle:
                handle.write('ep,done,reward,a0' + chr(10) + '0,0,0.0,0.1' + chr(10))
            alias = os.path.join(dirname, 'alias.csv')
            try:
                os.symlink(victim, alias)
            except OSError as error:
                self.skipTest(f'symlink creation is unavailable: {error}')
            with self.assertRaisesRegex(ValueError, 'regular|opened safely'):
                dataset.read_csv(alias, action_dim=1, goal_dim=0)
            with open(victim, encoding='utf-8') as handle:
                self.assertEqual(handle.read(), 'ep,done,reward,a0' + chr(10) + '0,0,0.0,0.1' + chr(10))
    def test_checkpoint_load_caps_bytes_and_refuses_links(self):
        import soul.soul as soul_module

        with tempfile.TemporaryDirectory() as dirname:
            path = os.path.join(dirname, 'big.pt')
            with open(path, 'wb') as handle:
                handle.write(b'x' * 64)
            with patch.object(soul_module, 'MAX_CHECKPOINT_BYTES', 8):
                with self.assertRaisesRegex(ValueError, 'byte limit'):
                    Soul(Config()).load(path)
            alias = os.path.join(dirname, 'alias.pt')
            try:
                os.symlink(path, alias)
            except OSError as error:
                self.skipTest(f'symlink creation is unavailable: {error}')
            with self.assertRaisesRegex(ValueError, 'regular|opened safely'):
                Soul(Config()).load(alias)
    def test_bounded_reader_keeps_binary_bytes_on_windows(self):
        from soul.safe_io import read_regular_bytes

        fd, path = tempfile.mkstemp(suffix='.bin')
        os.close(fd)
        try:
            payload = b'abc' + bytes([0x1A]) + b'def' + b'\r\n' + b'ghi'
            with open(path, 'wb') as handle:
                handle.write(payload)
            self.assertEqual(read_regular_bytes(path, max_bytes=64, label='blob'), payload)
        finally:
            os.remove(path)
    def test_memory_archive_load_caps_bytes_and_refuses_links(self):
        import soul.memory as memory_module
        from soul.memory import ArchiveMemory

        with tempfile.TemporaryDirectory() as dirname:
            path = os.path.join(dirname, 'archive.json')
            with open(path, 'wb') as handle:
                handle.write(b'x' * 64)
            with patch.object(memory_module, 'MAX_ARCHIVE_BYTES', 8):
                with self.assertRaisesRegex(ValueError, 'byte limit'):
                    ArchiveMemory().load(path)
            alias = os.path.join(dirname, 'alias.json')
            try:
                os.symlink(path, alias)
            except OSError as error:
                self.skipTest(f'symlink creation is unavailable: {error}')
            with self.assertRaisesRegex(ValueError, 'regular|opened safely'):
                ArchiveMemory().load(alias)
if __name__ == '__main__':
    unittest.main()


class ActionEffectTrainingTests(unittest.TestCase):
    def test_action_effect_head_is_trained(self):
        cfg = Config()
        cfg.obs.max_features = 4
        cfg.obs.text_buckets = 4
        cfg.model.hidden_dim = 8
        cfg.model.latent_dim = 4
        cfg.model.action_dim = 1
        cfg.model.goal_dim = 2
        cfg.train.imagine_horizon = 2
        cfg.train.seq_len = 3
        soul = Soul(cfg)
        rng = np.random.default_rng(7)
        seq = cfg.train.seq_len
        batch = {
            'values': rng.normal(size=(2, seq + 1, cfg.obs.max_features)).astype(np.float32),
            'masks': np.ones((2, seq + 1, cfg.obs.max_features), np.float32),
            'texts': np.zeros((2, seq + 1, cfg.obs.text_buckets), np.float32),
            'actions': rng.normal(size=(2, seq, cfg.model.action_dim)).astype(np.float32),
            'rewards': rng.normal(size=(2, seq)).astype(np.float32),
            'continues': np.ones((2, seq), np.float32),
            'goals': np.zeros((2, seq, cfg.model.goal_dim), np.float32),
            'valid': np.ones((2, seq), np.float32),
        }

        losses = soul.trainer.update(batch)

        self.assertTrue(np.isfinite(losses['action_effect']))
        head = soul.model.action_effect_head
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        norm = sum(float(p.grad.norm()) for p in head.parameters() if p.grad is not None)
        self.assertGreater(norm, 0.0)


class PlanningRewardAlignmentTests(unittest.TestCase):
    def test_planner_scores_with_hidden_after_action(self):
        cfg = Config()
        cfg.model.hidden_dim = 8
        cfg.model.latent_dim = 4
        cfg.model.action_dim = 1
        cfg.model.plan_samples = 3
        cfg.train.batch_size = 2
        cfg.train.seq_len = 4
        soul = Soul(cfg)
        obs = {'time': 0.0, 'x': 0.0, 'y': 0.0}
        zero = np.zeros(cfg.model.action_dim, dtype=np.float32)
        soul.predict_action_effect(obs, zero)  # initialise runtime state

        seen_hidden = []

        def prediction_spy(observation, action, **kwargs):
            latent, hidden = original_prediction(
                observation, action, return_hidden=True)
            seen_hidden.append(hidden.copy())
            return latent, hidden

        original_prediction = soul.predict_action_effect
        soul.predict_action_effect = prediction_spy

        seen_scores = []
        original_reward = soul.model.reward_head.forward

        def reward_spy(hidden, latent):
            seen_scores.append(hidden.detach().cpu().numpy().copy())
            return original_reward(hidden, latent)

        soul.model.reward_head.forward = reward_spy
        soul.plan_action(obs, n_samples=3)

        self.assertEqual(len(seen_hidden), len(seen_scores))
        self.assertGreater(len(seen_scores), 0)
        for hidden, scored in zip(seen_hidden, seen_scores):
            np.testing.assert_allclose(scored[0], hidden, rtol=1e-6, atol=1e-6)


class TrainingScheduleTests(unittest.TestCase):
    def test_each_batch_runs_two_updates(self):
        cfg = Config()
        self.assertEqual(cfg.train.updates_per_batch, 2)
        soul = Soul(cfg)

        calls = []
        soul.replay.sample = lambda *args, **kwargs: {}
        soul.trainer.update = lambda batch: calls.append(1) or {}

        self.assertIsNotNone(soul.learn(force=True))
        self.assertEqual(len(calls), 2)
