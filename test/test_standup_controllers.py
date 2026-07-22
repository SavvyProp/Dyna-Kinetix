"""Tests for configurable residual standup baseline controllers."""

import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynak-matplotlib")

import jax
import jax.numpy as jnp
import numpy as np

from dynak.standup.controllers import (
    UnderlyingControllerType,
    no_controller,
    resolve_underlying_controller,
)
from dynak.standup.residual_torque_env import make_residual_torque_env
from dynak.standup.stand_bb import (
    BB_TORQUE,
    sample_bang_bang_torque_nm,
    stand_bb,
    stand_bb_randomized,
)
from dynak.standup.stand_pd import (
    DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
    STANDUP_KD,
    STANDUP_KP,
    sample_controller_torque_noise_nm,
    sample_pd_gains,
    stand_pd,
    stand_pd_randomized,
)
from dynak.standup.stand_random import (
    BB_PARAMETER_KEY_TAG,
    CONTROLLER_PROBABILITIES,
    PD_PARAMETER_KEY_TAG,
    controller_switch_steps,
    get_random_controller_indices,
    stand_random,
)
from kinetix.environment import ObservationType
from kinetix.util import load_from_json_file


class TestStandupControllers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.static_env_params, cls.env_params = load_from_json_file(
            "l/standup_goal.json"
        )

    def test_nominal_bang_bang_keeps_constant_torque_near_target(self):
        # These rotations put the last two joints 0.07 rad from their targets.
        # PD therefore backs off while bang-bang remains at the shared limit.
        near_target = self.state.replace(
            polygon=self.state.polygon.replace(
                rotation=self.state.polygon.rotation.at[6].set(1.5)
            )
        )
        pd_torque = np.asarray(stand_pd(near_target, self.static_env_params))
        bang_bang_torque = np.asarray(stand_bb(near_target, self.static_env_params))

        np.testing.assert_allclose(pd_torque, [0.0, 0.35, -0.35], atol=1e-5)
        np.testing.assert_allclose(
            bang_bang_torque,
            [0.0, BB_TORQUE, -BB_TORQUE],
            atol=1e-6,
        )

    def test_no_controller_returns_zero_torque(self):
        torque = no_controller(
            self.state,
            self.static_env_params,
            jax.random.PRNGKey(0),
        )
        np.testing.assert_array_equal(np.asarray(torque), np.zeros(3))

    def test_controller_parameters_are_randomized_once_per_episode(self):
        first_key = jax.random.PRNGKey(1)
        second_key = jax.random.PRNGKey(2)

        near_target = self.state.replace(
            polygon=self.state.polygon.replace(
                rotation=self.state.polygon.rotation.at[6].set(1.5)
            )
        )

        for controller_name, direct_controller in (
            ("pd", stand_pd),
            ("bang_bang", stand_bb),
        ):
            with self.subTest(controller=controller_name):
                _, controller = resolve_underlying_controller(controller_name)
                first = controller(near_target, self.static_env_params, first_key)
                repeated = controller(
                    near_target,
                    self.static_env_params,
                    first_key,
                )
                second = controller(near_target, self.static_env_params, second_key)
                np.testing.assert_array_equal(np.asarray(first), np.asarray(repeated))
                self.assertFalse(np.allclose(np.asarray(first), np.asarray(second)))

                _, deterministic_controller = resolve_underlying_controller(
                    controller_name,
                    pd_gain_randomization_fraction=0.0,
                    bang_bang_torque_randomization_fraction=0.0,
                    controller_torque_noise_std_nm=0.0,
                )
                deterministic = deterministic_controller(
                    near_target,
                    self.static_env_params,
                    first_key,
                )
                expected = direct_controller(near_target, self.static_env_params)
                np.testing.assert_allclose(deterministic, expected, atol=1e-6)

    def test_controller_torque_noise_is_per_joint_and_per_step(self):
        episode_key = jax.random.PRNGKey(31)
        first = sample_controller_torque_noise_nm(
            episode_key,
            4,
            key_tag=999,
        )
        repeated = sample_controller_torque_noise_nm(
            episode_key,
            4,
            key_tag=999,
        )
        next_step = sample_controller_torque_noise_nm(
            episode_key,
            5,
            key_tag=999,
        )
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(next_step)))

        samples = jax.vmap(
            lambda key: sample_controller_torque_noise_nm(
                key,
                0,
                key_tag=999,
            )
        )(jax.random.split(jax.random.PRNGKey(32), 4096))
        samples = np.asarray(samples)
        np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.01)
        np.testing.assert_allclose(
            samples.std(axis=0),
            DEFAULT_CONTROLLER_TORQUE_NOISE_STD_NM,
            atol=0.01,
        )
        off_diagonal_covariance = np.cov(samples, rowvar=False) - np.diag(
            np.var(samples, axis=0, ddof=1)
        )
        np.testing.assert_allclose(off_diagonal_covariance, 0.0, atol=0.003)

    def test_randomized_parameters_stay_within_configured_range(self):
        randomization_fraction = 0.2
        episode_key = jax.random.PRNGKey(9)
        kp, kd = sample_pd_gains(episode_key, randomization_fraction)
        bb_torque = sample_bang_bang_torque_nm(
            episode_key,
            randomization_fraction,
        )

        np.testing.assert_array_less(
            np.asarray(STANDUP_KP * (1.0 - randomization_fraction)) - 1e-6,
            np.asarray(kp),
        )
        np.testing.assert_array_less(
            np.asarray(kp),
            np.asarray(STANDUP_KP * (1.0 + randomization_fraction)) + 1e-6,
        )
        np.testing.assert_array_less(
            np.asarray(STANDUP_KD * (1.0 - randomization_fraction)) - 1e-6,
            np.asarray(kd),
        )
        np.testing.assert_array_less(
            np.asarray(kd),
            np.asarray(STANDUP_KD * (1.0 + randomization_fraction)) + 1e-6,
        )
        self.assertTrue(
            np.all(np.asarray(bb_torque) >= BB_TORQUE * (1.0 - randomization_fraction))
        )
        self.assertTrue(
            np.all(np.asarray(bb_torque) <= BB_TORQUE * (1.0 + randomization_fraction))
        )

    def test_random_controller_switches_independently_per_joint(self):
        switch_key = jax.random.PRNGKey(17)
        steps_per_period = int(controller_switch_steps(self.static_env_params))
        np.testing.assert_allclose(
            np.asarray(CONTROLLER_PROBABILITIES),
            [0.2, 0.4, 0.4],
        )

        period_start = self.state.replace(timestep=jnp.asarray(0, dtype=jnp.int32))
        period_end = self.state.replace(
            timestep=jnp.asarray(steps_per_period - 1, dtype=jnp.int32)
        )
        start_choices = get_random_controller_indices(
            period_start,
            self.static_env_params,
            switch_key,
        )
        end_choices = get_random_controller_indices(
            period_end,
            self.static_env_params,
            switch_key,
        )
        np.testing.assert_array_equal(start_choices, end_choices)

        choices_by_period = []
        states_by_period = []
        for period_index in range(16):
            state = self.state.replace(
                timestep=jnp.asarray(
                    period_index * steps_per_period,
                    dtype=jnp.int32,
                )
            )
            states_by_period.append(state)
            choices_by_period.append(
                np.asarray(
                    get_random_controller_indices(
                        state,
                        self.static_env_params,
                        switch_key,
                    )
                )
            )

        mixed_period_index = next(
            index
            for index, choices in enumerate(choices_by_period)
            if np.unique(choices).size > 1
        )
        mixed_state = states_by_period[mixed_period_index]
        mixed_choices = choices_by_period[mixed_period_index]
        candidate_torques = np.stack(
            (
                np.zeros(3, dtype=np.float32),
                np.asarray(
                    stand_pd_randomized(
                        mixed_state,
                        self.static_env_params,
                        jax.random.fold_in(switch_key, PD_PARAMETER_KEY_TAG),
                    )
                ),
                np.asarray(
                    stand_bb_randomized(
                        mixed_state,
                        self.static_env_params,
                        jax.random.fold_in(switch_key, BB_PARAMETER_KEY_TAG),
                    )
                ),
            )
        )
        expected = candidate_torques[mixed_choices, np.arange(3)]
        actual = stand_random(
            mixed_state,
            self.static_env_params,
            switch_key,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_controller_names_and_aliases_resolve(self):
        expected = {
            "pd": "pd",
            "stand_pd": "pd",
            "bb": "bang_bang",
            "stand-bb": "bang_bang",
            "random": "random",
            "random-switch": "random",
            "mixed": "random",
            "none": "none",
            "no_controller": "none",
        }
        for alias, canonical_name in expected.items():
            with self.subTest(alias=alias):
                name, controller = resolve_underlying_controller(alias)
                self.assertEqual(name, canonical_name)
                self.assertTrue(callable(controller))

        with self.assertRaisesRegex(ValueError, "Unknown underlying controller"):
            resolve_underlying_controller("not-a-controller")
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            resolve_underlying_controller(
                "pd",
                controller_torque_noise_std_nm=-0.1,
            )

    def test_environment_accepts_builtins_and_custom_callable(self):
        for controller_type in UnderlyingControllerType:
            with self.subTest(controller=controller_type.value):
                env = make_residual_torque_env(
                    observation_type=ObservationType.SYMBOLIC_FLAT,
                    reset_fn=lambda _rng: self.state,
                    env_params=self.env_params,
                    static_env_params=self.static_env_params,
                    auto_reset=False,
                    underlying_controller=controller_type,
                )
                self.assertEqual(
                    env.underlying_controller_name,
                    controller_type.value,
                )

        def constant_controller(state, static_env_params, episode_key):
            del state, static_env_params, episode_key
            return jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

        env = make_residual_torque_env(
            observation_type=ObservationType.SYMBOLIC_FLAT,
            reset_fn=lambda _rng: self.state,
            env_params=self.env_params,
            static_env_params=self.static_env_params,
            auto_reset=False,
            underlying_controller=constant_controller,
        )
        _, residual_state = env.reset_env(
            jax.random.PRNGKey(1),
            self.env_params,
            self.state,
        )
        torque = env.compute_underlying_torque_nm(residual_state)
        self.assertEqual(env.underlying_controller_name, "constant_controller")
        np.testing.assert_array_equal(np.asarray(torque), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
