"""
GFlowNet
TODO:
    - Seeds
"""
import copy
import time
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count
from pathlib import Path

from comet_ml import Experiment
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from gflownetenv import Buffer
from aptamers import AptamerSeq
from grid import Grid
from oracle import numbers2letters, Oracle
from utils import get_config, namespace2dict, numpy2python, add_bool_arg

# Float and Long tensors
_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    args2config = {}
    # YAML config
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="YAML configuration file",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
    # General
    parser.add_argument("--workdir", default=None, type=str)
    args2config.update({"workdir": ["workdir"]})
    parser.add_argument("--overwrite_workdir", action="store_true", default=False)
    args2config.update({"overwrite_workdir": ["overwrite_workdir"]})
    parser.add_argument("--device", default="cpu", type=str)
    args2config.update({"device": ["gflownet", "device"]})
    parser.add_argument("--progress", action="store_true")
    args2config.update({"progress": ["gflownet", "progress"]})
    parser.add_argument("--no_lightweight", action="store_true")
    args2config.update({"no_lightweight": ["no_lightweight"]})
    parser.add_argument("--debug", action="store_true")
    args2config.update({"debug": ["debug"]})
    parser.add_argument("--model_ckpt", default=None, type=str)
    args2config.update({"model_ckpt": ["gflownet", "model_ckpt"]})
    parser.add_argument("--reload_ckpt", action="store_true")
    args2config.update({"reload_ckpt": ["gflownet", "reload_ckpt"]})
    parser.add_argument("--ckpt_period", default=None, type=int)
    args2config.update({"ckpt_period": ["gflownet", "ckpt_period"]})
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="Seed for random number generator",
    )
    args2config.update({"rng_seed": ["seeds", "gflownet"]})
    # Oracle
    parser.add_argument(
        "--oracle_seed",
        type=int,
        default=0,
        help="Seed for oracle",
    )
    args2config.update({"oracle_seed": ["seeds", "oracle"]})
    parser = add_bool_arg(parser, "nupack_energy_reweighting", default=False)
    args2config.update(
        {"nupack_energy_reweighting": ["dataset", "nupack_energy_reweighting"]}
    )
    parser.add_argument(
        "--nupack_target_motif",
        type=str,
        default=".....(((((.......))))).....",
        help="if using 'nupack motif' oracle, return value is the binary distance to this fold, must be <= max sequence length",
    )
    args2config.update({"nupack_target_motif": ["dataset", "nupack_target_motif"]})
    # Training hyperparameters
    parser.add_argument(
        "--loss", default="flowmatch", type=str, help="flowmatch | trajectorybalance/tb"
    )
    args2config.update({"loss": ["gflownet", "loss"]})
    parser.add_argument(
        "--lr_z_mult",
        default=10,
        type=int,
        help="Multiplicative factor of the Z learning rate",
    )
    args2config.update({"lr_z_mult": ["gflownet", "lr_z_mult"]})
    parser.add_argument(
        "--early_stopping",
        default=0.01,
        help="Threshold loss for early stopping",
        type=float,
    )
    args2config.update({"early_stopping": ["gflownet", "early_stopping"]})
    parser.add_argument(
        "--ema_alpha",
        default=0.5,
        help="alpha coefficient for exponential moving average",
        type=float,
    )
    args2config.update({"ema_alpha": ["gflownet", "ema_alpha"]})
    parser.add_argument(
        "--learning_rate", default=1e-4, help="Learning rate", type=float
    )
    args2config.update({"learning_rate": ["gflownet", "learning_rate"]})
    parser.add_argument(
        "--lr_decay_period", default=1e6, help="Learning rate decay period", type=int
    )
    args2config.update({"lr_decay_period": ["gflownet", "lr", "decay_period"]})
    parser.add_argument(
        "--lr_decay_gamma", default=0.5, help="Learning rate decay gamma", type=float
    )
    args2config.update({"lr_decay_gamma": ["gflownet", "lr", "decay_gamma"]})
    parser.add_argument("--opt", default="adam", type=str)
    args2config.update({"opt": ["gflownet", "opt"]})
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    args2config.update({"adam_beta1": ["gflownet", "adam_beta1"]})
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    args2config.update({"adam_beta2": ["gflownet", "adam_beta2"]})
    parser.add_argument(
        "--reward_beta",
        default=1,
        type=float,
        help="Initial beta for exponential reward scaling",
    )
    args2config.update({"reward_beta": ["gflownet", "reward_beta"]})
    parser.add_argument(
        "--reward_norm",
        default=1.0,
        type=float,
        help="Factor for the reward normalization",
    )
    args2config.update({"reward_norm": ["gflownet", "reward_norm"]})
    parser.add_argument(
        "--reward_norm_std_mult",
        default=0.0,
        type=float,
        help="Multiplier of the standard deviation for the reward normalization",
    )
    args2config.update({"reward_norm_std_mult": ["gflownet", "reward_norm_std_mult"]})
    parser.add_argument(
        "--reward_func",
        default=1,
        type=float,
        help="Function for rewards transformation: power or boltzmann",
    )
    args2config.update({"reward_func": ["gflownet", "reward_func"]})
    parser.add_argument("--momentum", default=0.9, type=float)
    args2config.update({"momentum": ["gflownet", "momentum"]})
    parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
    args2config.update({"mbsize": ["gflownet", "mbsize"]})
    parser.add_argument("--train_to_sample_ratio", default=1, type=float)
    args2config.update({"train_to_sample_ratio": ["gflownet", "train_to_sample_ratio"]})
    parser.add_argument("--n_hid", default=256, type=int)
    args2config.update({"n_hid": ["gflownet", "n_hid"]})
    parser.add_argument("--n_layers", default=2, type=int)
    args2config.update({"n_layers": ["gflownet", "n_layers"]})
    parser.add_argument("--n_train_steps", default=20000, type=int)
    args2config.update({"n_train_steps": ["gflownet", "n_iter"]})
    parser.add_argument(
        "--num_empirical_loss",
        default=200000,
        type=int,
        help="Number of samples used to compute the empirical distribution loss",
    )
    args2config.update({"num_empirical_loss": ["gflownet", "num_empirical_loss"]})
    parser.add_argument("--clip_grad_norm", default=0.0, type=float)
    args2config.update({"clip_grad_norm": ["gflownet", "clip_grad_norm"]})
    parser.add_argument("--temperature_logits", default=0.0, type=float)
    args2config.update({"temperature_logits": ["gflownet", "temperature_logits"]})
    parser.add_argument("--pct_batch_empirical", default=1.0, type=float)
    args2config.update({"pct_batch_empirical": ["gflownet", "pct_batch_empirical"]})
    parser.add_argument("--replay_capacity", default=0, type=int)
    args2config.update({"replay_capacity": ["gflownet", "replay_capacity"]})
    # Environment
    parser.add_argument("--env_id", default="aptamers")
    args2config.update({"env_id": ["gflownet", "env_id"]})
    parser.add_argument("--func", default="arbitrary_i")
    args2config.update({"func": ["gflownet", "func"]})
    parser.add_argument(
        "--max_seq_length",
        default=40,
        help="Maximum number of episodes; maximum sequence length",
        type=int,
    )
    args2config.update({"max_seq_length": ["gflownet", "max_seq_length"]})
    parser.add_argument(
        "--min_seq_length",
        default=1,
        help="Minimum sequence length",
        type=int,
    )
    args2config.update({"min_seq_length": ["gflownet", "min_seq_length"]})
    parser.add_argument("--nalphabet", default=4, type=int)
    args2config.update({"nalphabet": ["gflownet", "nalphabet"]})
    parser.add_argument("--min_word_len", default=1, type=int)
    args2config.update({"min_word_len": ["gflownet", "min_word_len"]})
    parser.add_argument("--max_word_len", default=1, type=int)
    args2config.update({"max_word_len": ["gflownet", "max_word_len"]})
    args2config.update({"learning_rate": ["gflownet", "learning_rate"]})
    # Sampling
    parser.add_argument("--bootstrap_tau", default=0.0, type=float)
    args2config.update({"bootstrap_tau": ["gflownet", "bootstrap_tau"]})
    parser.add_argument("--batch_reward", action="store_true")
    args2config.update({"batch_reward": ["gflownet", "batch_reward"]})
    # Train set
    parser.add_argument("--train_set_path", default=None, type=str)
    args2config.update({"train_set_path": ["gflownet", "train", "path"]})
    parser.add_argument("--ntrain", default=10000, type=int)
    args2config.update({"ntrain": ["gflownet", "train", "n"]})
    parser.add_argument("--train_set_seed", default=167, type=int)
    args2config.update({"train_set_seed": ["gflownet", "train", "seed"]})
    parser.add_argument("--train_output", default=None, type=str)
    args2config.update({"train_output": ["gflownet", "train", "output"]})
    # Test set
    parser.add_argument("--test_set_path", default=None, type=str)
    args2config.update({"test_set_path": ["gflownet", "test", "path"]})
    parser.add_argument("--test_set_base", default=None, type=str)
    args2config.update({"test_set_base": ["gflownet", "test", "base"]})
    parser.add_argument("--test_set_seed", default=167, type=int)
    args2config.update({"test_set_seed": ["gflownet", "test", "seed"]})
    parser.add_argument("--ntest", default=10000, type=int)
    args2config.update({"ntest": ["gflownet", "test", "n"]})
    parser.add_argument("--test_output", default=None, type=str)
    args2config.update({"test_output": ["gflownet", "test", "output"]})
    parser.add_argument("--test_period", default=500, type=int)
    args2config.update({"test_period": ["gflownet", "test", "period"]})
    # Oracle metrics
    parser.add_argument("--oracle_period", default=500, type=int)
    args2config.update({"oracle_period": ["gflownet", "oracle", "period"]})
    parser.add_argument("--oracle_nsamples", default=500, type=int)
    args2config.update({"oracle_nsamples": ["gflownet", "oracle", "nsamples"]})
    parser.add_argument(
        "--oracle_k",
        default=[1, 10, 100],
        nargs="*",
        type=int,
        help="List of K, for Top-K",
    )
    args2config.update({"oracle_k": ["gflownet", "oracle", "k"]})
    # Comet
    parser.add_argument("--comet_project", default=None, type=str)
    args2config.update({"comet_project": ["gflownet", "comet", "project"]})
    parser.add_argument(
        "-t", "--tags", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    args2config.update({"tags": ["gflownet", "comet", "tags"]})
    parser.add_argument("--no_comet", action="store_true")
    args2config.update({"no_comet": ["gflownet", "comet", "skip"]})
    parser.add_argument("--no_log_times", action="store_true")
    args2config.update({"no_log_times": ["gflownet", "no_log_times"]})
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Sequences to sample",
    )
    args2config.update({"n_samples": ["gflownet", "n_samples"]})

    return parser, args2config


def process_config(config):
    if "score" not in config.gflownet.test or "nupack" in config.gflownet.test.score:
        config.gflownet.test.score = config.gflownet.func.replace("nupack ", "")
    return config


def set_device(dev):
    _dev[0] = dev


class GFlowNetAgent:
    def __init__(
        self,
        args,
        comet=None,
        proxy=None,
        al_iter=-1,
        data_path=None,
        sample_only=False,
    ):
        # Misc
        self.rng = np.random.default_rng(args.seeds.gflownet)
        self.debug = args.debug
        self.device_torch = torch.device(args.gflownet.device)
        self.device = self.device_torch
        set_device(self.device_torch)
        if args.gflownet.loss in ["flowmatch"]:
            self.loss = "flowmatch"
            self.Z = None
        elif args.gflownet.loss in ["trajectorybalance", "tb"]:
            self.loss = "trajectorybalance"
            self.Z = nn.Parameter(torch.ones(64) * 150.0 / 64)
        else:
            print("Unkown loss. Using flowmatch as default")
            self.loss = "flowmatch"
            self.Z = None
        if not sample_only:
            self.loss_eps = torch.tensor(float(1e-5)).to(self.device)
        self.lightweight = not args.no_lightweight
        self.tau = args.gflownet.bootstrap_tau
        self.ema_alpha = args.gflownet.ema_alpha
        self.early_stopping = args.gflownet.early_stopping
        self.reward_norm = np.abs(args.gflownet.reward_norm)
        self.reward_norm_std_mult = args.gflownet.reward_norm_std_mult
        self.reward_beta = args.gflownet.reward_beta
        self.reward_func = args.gflownet.reward_func
        if al_iter >= 0:
            self.al_iter = "_iter{}".format(al_iter)
        else:
            self.al_iter = ""
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.batch_reward = args.gflownet.batch_reward
        self.env_id = args.gflownet.env_id.lower()
        # Oracle
        if self.env_id == "aptamers":
            self.oracle = Oracle(
                oracle=args.gflownet.func,
                seed=args.seeds.oracle,
                seq_len=args.gflownet.max_seq_length,
                dict_size=args.gflownet.nalphabet,
                min_len=args.gflownet.min_seq_length,
                max_len=args.gflownet.max_seq_length,
                energy_weight=args.dataset.nupack_energy_reweighting,
                nupack_target_motif=args.dataset.nupack_target_motif,
            )
        elif self.env_id == "grid":
            self.oracle = None
        else:
            raise NotImplemented
        # Environment
        if self.env_id == "aptamers":
            self.env = AptamerSeq(
                args.gflownet.max_seq_length,
                args.gflownet.min_seq_length,
                args.gflownet.nalphabet,
                args.gflownet.min_word_len,
                args.gflownet.max_word_len,
                proxy=proxy,
                reward_beta=self.reward_beta,
                reward_norm=self.reward_norm,
                reward_func=self.reward_func,
                oracle_func=self.oracle.score,
                debug=self.debug,
            )
        elif self.env_id == "grid":
            self.env = Grid(
                reward_func=self.reward_func,
                oracle_func=args.gflownet.func,
            )
        else:
            raise NotImplemented
        self.buffer = Buffer(self.env, replay_capacity=args.gflownet.replay_capacity)
        # Comet
        if (
            args.gflownet.comet.project
            and not args.gflownet.comet.skip
            and not sample_only
        ):
            self.comet = Experiment(
                project_name=args.gflownet.comet.project, display_summary_level=0
            )
            if args.gflownet.comet.tags:
                if isinstance(args.gflownet.comet.tags, list):
                    self.comet.add_tags(args.gflownet.comet.tags)
                else:
                    self.comet.add_tag(args.gflownet.comet.tags)
            self.comet.log_parameters(vars(args))
            if "workdir" in args and Path(args.workdir).exists():
                with open(Path(args.workdir) / "comet.url", "w") as f:
                    f.write(self.comet.url + "\n")
        else:
            if isinstance(comet, Experiment):
                self.comet = comet
            else:
                self.comet = None
        self.no_log_times = args.gflownet.no_log_times
        # Make train and test sets
        if not sample_only:
            self.buffer.make_train_test(
                data_path,
                args.gflownet.train.path,
                args.gflownet.test.path,
                self.oracle,
                args,
            )
        self.test_period = args.gflownet.test.period
        if self.test_period in [None, -1]:
            self.test_period = np.inf
        # Train set statistics
        if self.buffer.train is not None:
            min_energies_tr = self.buffer.train["energies"].min()
            max_energies_tr = self.buffer.train["energies"].max()
            mean_energies_tr = self.buffer.train["energies"].mean()
            std_energies_tr = self.buffer.train["energies"].std()
            energies_tr_norm = (
                self.buffer.train["energies"].values - mean_energies_tr
            ) / std_energies_tr
            max_norm_energies_tr = np.max(energies_tr_norm)
            self.energies_stats_tr = [
                min_energies_tr,
                max_energies_tr,
                mean_energies_tr,
                std_energies_tr,
                max_norm_energies_tr,
            ]
            self.env.set_energies_stats(self.energies_stats_tr)
            print("\nTrain data")
            print(f"\tAverage score: {mean_energies_tr}")
            print(f"\tStd score: {std_energies_tr}")
            print(f"\tMin score: {min_energies_tr}")
            print(f"\tMax score: {max_energies_tr}")
        else:
            self.energies_stats_tr = None
        if self.reward_norm_std_mult > 0 and self.energies_stats_tr is not None:
            self.reward_norm = self.reward_norm_std_mult * self.energies_stats_tr[3]
            self.env.set_reward_norm(self.reward_norm)
        # Test set statistics
        if self.buffer.test is not None:
            print("\nTest data")
            print(f"\tAverage score: {self.buffer.test['energies'].mean()}")
            print(f"\tStd score: {self.buffer.test['energies'].std()}")
            print(f"\tMin score: {self.buffer.test['energies'].min()}")
            print(f"\tMax score: {self.buffer.test['energies'].max()}")
        # Model
        self.model = make_mlp(
            [self.env.obs_dim]
            + [args.gflownet.n_hid] * args.gflownet.n_layers
            + [len(self.env.action_space) + 1]
        )
        self.reload_ckpt = args.gflownet.reload_ckpt
        if args.gflownet.model_ckpt:
            if "workdir" in args and Path(args.workdir).exists():
                if (Path(args.workdir) / "ckpts").exists():
                    self.model_path = (
                        Path(args.workdir) / "ckpts" / args.gflownet.model_ckpt
                    )
                else:
                    self.model_path = Path(args.workdir) / args.gflownet.model_ckpt
            else:
                self.model_path = args.gflownet.model_ckpt
            if self.model_path.exists() and self.reload_ckpt:
                self.model.load_state_dict(torch.load(self.model_path))
                print("Reloaded GFN Model Checkpoint")
        else:
            self.model_path = None
        self.ckpt_period = args.gflownet.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        self.model.to(self.device_torch)
        self.target = copy.deepcopy(self.model)
        # Training
        self.opt, self.lr_scheduler = make_opt(self.parameters(), self.Z, args)
        self.n_train_steps = args.gflownet.n_iter
        self.mbsize = args.gflownet.mbsize
        self.mask_invalid_actions = True
        self.progress = args.gflownet.progress
        self.clip_grad_norm = args.gflownet.clip_grad_norm
        self.num_empirical_loss = args.gflownet.num_empirical_loss
        self.ttsr = max(int(args.gflownet.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / args.gflownet.train_to_sample_ratio), 1)
        self.temperature_logits = args.gflownet.temperature_logits
        self.pct_batch_empirical = args.gflownet.pct_batch_empirical
        # Oracle metrics
        self.oracle_period = args.gflownet.oracle.period
        self.oracle_nsamples = args.gflownet.oracle.nsamples
        self.oracle_k = args.gflownet.oracle.k

    def parameters(self):
        return self.model.parameters()

    def forward_sample(self, envs, times, policy="model", model=None, temperature=1.0):
        """
        Performs a forward action on each environment of a list.

        Args
        ----
        env : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        policy : string
            - model: uses self.model to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if policy="model"

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        if not isinstance(envs, list):
            envs = [envs]
        states = [env.state2obs() for env in envs]
        mask_invalid_actions = [env.get_mask_invalid_actions() for env in envs]
        random_action = self.rng.uniform()
        t0_a_model = time.time()
        if policy == "model":
            with torch.no_grad():
                action_logits = model(tf(states))
            action_logits /= temperature
        elif policy == "uniform":
            action_logits = tf(np.zeros(len(states)), len(self.env.action_space) + 1)
        else:
            raise NotImplemented
        if self.mask_invalid_actions:
            action_logits[torch.tensor(mask_invalid_actions)] = -1000
        if all(torch.isfinite(action_logits).flatten()):
            actions = Categorical(logits=action_logits).sample()
        else:
            if self.debug:
                raise ValueError("Action could not be sampled from model!")
        t1_a_model = time.time()
        times["actions_model"] += t1_a_model - t0_a_model
        assert len(envs) == actions.shape[0]
        # Execute actions
        _, _, valids = zip(*[env.step(action) for env, action in zip(envs, actions)])
        return envs, actions, valids

    def backward_sample(
        self, env, policy="model", model=None, temperature=1.0, done=False
    ):
        """
        Performs a backward action on one environment.

        Args
        ----
        env : GFlowNetEnv or derived
            An instance of the environment

        policy : string
            - model: uses the current policy to obtain the sampling probabilities.
            - uniform: samples uniformly from the parents of the current state.

        model : torch model
            Model to use as policy if policy="model"

        temperature : float
            Temperature to adjust the logits by logits /= temperature

        done : bool
            If True, it will sample eos action
        """
        parents, parents_a = env.get_parents(env.state, done)
        if policy == "model":
            with torch.no_grad():
                action_logits = model(tf(parents))[
                    torch.arange(len(parents)), parents_a
                ]
            action_logits /= temperature
            if all(torch.isfinite(action_logits).flatten()):
                action_idx = Categorical(logits=action_logits).sample().item()
            else:
                if self.debug:
                    raise ValueError("Action could not be sampled from model!")
        elif policy == "uniform":
            action_idx = self.rng.integers(low=0, high=len(parents_a))
        else:
            raise NotImplemented
        action = parents_a[action_idx]
        env.set_state(env.obs2state((parents)[action_idx]), done=False)
        return env, env.state, action, parents, parents_a

    def sample_batch(
        self, envs, n_samples=None, train=True, model=None, progress=False
    ):
        """
        Builds a batch of data

        if train == True:
            Each item in the batch is a list of 7 elements (all tensors):
                - [0] the state, as state2obs(state)
                - [1] the action
                - [2] reward of the state
                - [3] all parents of the state
                - [4] actions that lead to the state from each parent
                - [5] done [True, False]
                - [6] path id: identifies each path
                - [7] state id: identifies each state within a path
        else:
            Each item in the batch is a list of 1 element:
                - [0] the states (state)

        Args
        ----
        """
        times = {
            "all": 0.0,
            "actions_model": 0.0,
            "actions_envs": 0.0,
            "rewards": 0.0,
        }
        t0_all = time.time()
        batch = []
        if model is None:
            model = self.model
        if isinstance(envs, list):
            envs = [env.reset(idx) for idx, env in enumerate(envs)]
        elif n_samples is not None and n_samples > 0:
            envs = [copy.deepcopy(envs).reset(idx) for idx in range(n_samples)]
        else:
            return None, None
        # Offline trajectories
        # TODO: Replay Buffer
        if train:
            n_empirical = int(self.pct_batch_empirical * len(envs))
            for env in envs[:n_empirical]:
                readable = self.rng.permutation(self.buffer.train.samples.values)[0]
                env = env.set_state(env.readable2state(readable), done=True)
                action = env.eos
                parents = [env.state2obs(env.state)]
                parents_a = [action]
                n_actions = 0
                while len(env.state) > 0:
                    batch.append(
                        [
                            tf([env.state2obs(env.state)]),
                            tl([action]),
                            env.state,
                            tf(parents),
                            tl(parents_a),
                            env.done,
                            tl([env.id] * len(parents)),
                            tl([n_actions]),
                        ]
                    )
                    # Backward sampling
                    env, state, action, parents, parents_a = self.backward_sample(
                        env,
                        policy="model",
                        model=model,
                        temperature=self.temperature_logits,
                    )
                    n_actions += 1
            envs = envs[n_empirical:]
        # Policy trajectories
        while envs:
            # Forward sampling
            if train is False:
                envs, actions, valids = self.forward_sample(
                    envs, times, policy="model", model=model, temperature=1.0
                )
            else:
                envs, actions, valids = self.forward_sample(
                    envs,
                    times,
                    policy="model",
                    model=model,
                    temperature=self.temperature_logits,
                )
            t0_a_envs = time.time()
            # Add to batch
            for env, action, valid in zip(envs, actions, valids):
                if valid:
                    parents, parents_a = env.get_parents()
                    if train:
                        batch.append(
                            [
                                tf([env.state2obs()]),
                                tl([action]),
                                env.state,
                                tf(parents),
                                tl(parents_a),
                                env.done,
                                tl([env.id] * len(parents)),
                                tl([env.n_actions - 1]),
                            ]
                        )
                    else:
                        if env.done:
                            batch.append(env.state)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
            if progress and n_samples is not None:
                print(f"{n_samples - len(envs)}/{n_samples} done")
        if train:
            # Compute rewards
            obs, actions, states, parents, parents_a, done, path_id, state_id = zip(
                *batch
            )
            t0_rewards = time.time()
            rewards = env.reward_batch(states, done)
            t1_rewards = time.time()
            times["rewards"] += t1_rewards - t0_rewards
            rewards = [tf([r]) for r in rewards]
            done = [tl([d]) for d in done]
            batch = list(
                zip(obs, actions, rewards, parents, parents_a, done, path_id, state_id)
            )
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return batch, times

    def flowmatch_loss(self, it, batch):
        """
        Computes the loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each state in the batch.

        Returns
        -------
        loss : float
            Loss, as per Equation 12 of https://arxiv.org/abs/2106.04399v1

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        loginf = tf([1000])
        batch_idxs = tl(
            sum(
                [
                    [i] * len(parents)
                    for i, (_, _, _, parents, _, _, _, _) in enumerate(batch)
                ],
                [],
            )
        )
        sp, _, r, parents, actions, done, _, _ = map(torch.cat, zip(*batch))
        # Sanity check if negative rewards
        if self.debug and torch.any(r < 0):
            neg_r_idx = torch.where(r < 0)[0].tolist()
            for idx in neg_r_idx:
                obs = sp[idx].tolist()
                state = list(self.env.obs2state(obs))
                state_oracle = self.env.state2oracle([state])
                output_proxy = self.env.proxy(state_oracle)
                reward = self.env.proxy2reward(output_proxy)
                import ipdb

                ipdb.set_trace()

        # Q(s,a)
        parents_Qsa = self.model(parents)[torch.arange(parents.shape[0]), actions]

        # log(eps + exp(log(Q(s,a)))) : qsa
        in_flow = torch.log(
            self.loss_eps
            + tf(torch.zeros((sp.shape[0],))).index_add_(
                0, batch_idxs, torch.exp(parents_Qsa)
            )
        )
        # the following with work if autoregressive
        #         in_flow = torch.logaddexp(parents_Qsa[batch_idxs], torch.log(self.loss_eps))
        if self.tau > 0:
            with torch.no_grad():
                next_q = self.target(sp)
        else:
            # TODO: potentially mask invalid actions next_q
            next_q = self.model(sp)
        qsp = torch.logsumexp(next_q, 1)
        # qsp: qsp if not done; -loginf if done
        qsp = qsp * (1 - done) - loginf * done
        out_flow = torch.logaddexp(torch.log(r + self.loss_eps), qsp)
        loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (
                done.sum() + 1e-20
            )
            flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / (
                (1 - done).sum() + 1e-20
            )

        if self.tau > 0:
            for a, b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(1 - self.tau).add_(self.tau * a)

        return loss, term_loss, flow_loss

    def trajectorybalance_loss(self, it, batch):
        """
        Computes the trajectory balance loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each state in the batch.

        Returns
        -------
        loss : float

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        # Unpack batch
        _, _, rewards, parents, actions, done, path_id_parents, _ = zip(*batch)
        path_id = torch.cat([el[:1] for el in path_id_parents])
        rewards, parents, actions, done, path_id_parents = map(
            torch.cat, [rewards, parents, actions, done, path_id_parents]
        )
        # Log probs of each (s, a)
        logprobs = self.logsoftmax(self.model(parents))[
            torch.arange(parents.shape[0]), actions
        ]
        # Sum of log probs
        sumlogprobs = tf(
            torch.zeros(len(torch.unique(path_id, sorted=True)))
        ).index_add_(0, path_id_parents, logprobs)
        # Sort rewards of done states by ascending path id
        rewards = rewards[done.eq(1)][torch.argsort(path_id[done.eq(1)])]
        # Trajectory balance loss
        loss = (self.Z.sum() + sumlogprobs - torch.log((rewards))).pow(2).mean()
        return loss, loss, loss

    def unpack_terminal_states(self, batch):
        paths = [[] for _ in range(self.mbsize)]
        states = [None] * self.mbsize
        rewards = [None] * self.mbsize
        #         state_ids = [[-1] for _ in range(self.mbsize)]
        for el in batch:
            path_id = el[6][:1].item()
            state_id = el[7][:1].item()
            #             assert state_ids[path_id][-1] + 1 == state_id
            #             state_ids[path_id].append(state_id)
            paths[path_id].append(el[1][0].item())
            if bool(el[5].item()):
                states[path_id] = tuple(self.env.obs2state(el[0][0].tolist()))
                rewards[path_id] = el[2][0].item()
        paths = [tuple(el) for el in paths]
        return states, paths, rewards

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None
        loss_flow_ema = None
        # Generate list of environments
        envs = [copy.deepcopy(self.env).reset() for _ in range(self.mbsize)]
        # Train loop
        for it in tqdm(range(self.n_train_steps + 1), disable=not self.progress):
            t0_iter = time.time()
            data = []
            for j in range(self.sttr):
                batch, times = self.sample_batch(envs)
                data += batch
            for j in range(self.ttsr):
                if self.loss == "flowmatch":
                    losses = self.flowmatch_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                elif self.loss == "trajectorybalance":
                    losses = self.trajectorybalance_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                else:
                    print("Unknown loss!")
                if not all([torch.isfinite(loss) for loss in losses]):
                    if self.debug:
                        print("Loss is not finite - skipping iteration")
                    if len(all_losses) > 0:
                        all_losses.append([loss for loss in all_losses[-1]])
                else:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            # Buffer
            states_term, paths_term, rewards = self.unpack_terminal_states(batch)
            proxy_vals = self.env.reward2proxy(rewards)
            self.buffer.add(states_term, paths_term, rewards, proxy_vals, it)
            self.buffer.add(
                states_term, paths_term, rewards, proxy_vals, it, buffer="replay"
            )
            # Log
            idx_best = np.argmax(rewards)
            state_best = "".join(self.env.state2readable(states_term[idx_best]))
            if self.lightweight:
                all_losses = all_losses[-100:]
                all_visited = states_term

            else:
                all_visited.extend(states_term)
            if self.comet:
                self.comet.log_text(
                    state_best + " / proxy: {}".format(proxy_vals[idx_best]), step=it
                )
                self.comet.log_metrics(
                    dict(
                        zip(
                            [
                                "mean_reward{}".format(self.al_iter),
                                "max_reward{}".format(self.al_iter),
                                "mean_proxy{}".format(self.al_iter),
                                "min_proxy{}".format(self.al_iter),
                                "max_proxy{}".format(self.al_iter),
                                "mean_seq_length{}".format(self.al_iter),
                                "batch_size{}".format(self.al_iter),
                            ],
                            [
                                np.mean(rewards),
                                np.max(rewards),
                                np.mean(proxy_vals),
                                np.min(proxy_vals),
                                np.max(proxy_vals),
                                np.mean([len(state) for state in states_term]),
                                len(data),
                            ],
                        )
                    ),
                    step=it,
                )
            # Test set metrics
            if not it % self.test_period and self.buffer.test is not None:
                data_logq = []
                times.update(
                    {
                        "test_paths": 0.0,
                        "test_logq": 0.0,
                    }
                )
                # TODO: this could be done just once and store it
                for statestr, score in tqdm(
                    zip(self.buffer.test.samples, self.buffer.test["energies"]),
                    disable=self.test_period < 10,
                ):
                    t0_test_path = time.time()
                    path_list, actions = self.env.get_paths(
                        [[self.env.readable2state(statestr)]],
                        [[self.env.eos]],
                    )
                    t1_test_path = time.time()
                    times["test_paths"] += t1_test_path - t0_test_path
                    t0_test_logq = time.time()
                    data_logq.append(logq(path_list, actions, self.model, self.env))
                    t1_test_logq = time.time()
                    times["test_logq"] += t1_test_logq - t0_test_logq
                corr = np.corrcoef(data_logq, self.buffer.test["energies"])
                if self.comet:
                    self.comet.log_metrics(
                        dict(
                            zip(
                                [
                                    "test_corr_logq_score{}".format(self.al_iter),
                                    "test_mean_logq{}".format(self.al_iter),
                                ],
                                [
                                    corr[0, 1],
                                    np.mean(data_logq),
                                ],
                            )
                        ),
                        step=it,
                    )
            # Oracle metrics (for monitoring)
            if not it % self.oracle_period and self.debug:
                oracle_batch, oracle_times = self.sample_batch(
                    self.env, self.oracle_nsamples, train=False
                )
                oracle_dict, oracle_times = batch2dict(
                    oracle_batch, self.env, get_uncertainties=False
                )
                energies = oracle_dict["energies"]
                energies_sorted = np.sort(energies)
                dict_topk = {}
                for k in self.oracle_k:
                    mean_topk = np.mean(energies_sorted[:k])
                    dict_topk.update(
                        {"oracle_mean_top{}{}".format(k, self.al_iter): mean_topk}
                    )
                    if self.comet:
                        self.comet.log_metrics(dict_topk)
            if not it % 100:
                if not self.lightweight:
                    l1_error, kl_div = empirical_distribution_error(
                        self.env, all_visited[-self.num_empirical_loss :]
                    )
                else:
                    l1_error, kl_div = 1, 100
                if self.progress:
                    if self.debug:
                        print("Empirical L1 distance", l1_error, "KL", kl_div)
                        if len(all_losses):
                            print(
                                *[
                                    f"{np.mean([i[j] for i in all_losses[-100:]]):.5f}"
                                    for j in range(len(all_losses[0]))
                                ]
                            )
                if self.comet:
                    self.comet.log_metrics(
                        dict(
                            zip(
                                [
                                    "loss{}".format(self.al_iter),
                                    "term_loss{}".format(self.al_iter),
                                    "flow_loss{}".format(self.al_iter),
                                    "l1{}".format(self.al_iter),
                                    "kl{}".format(self.al_iter),
                                ],
                                [loss.item() for loss in losses] + [l1_error, kl_div],
                            )
                        ),
                        step=it,
                    )
                    if not self.lightweight:
                        self.comet.log_metric(
                            "unique_states{}".format(self.al_iter),
                            np.unique(all_visited).shape[0],
                            step=it,
                        )
            # Save intermediate model
            if not it % self.ckpt_period and self.model_path:
                path = self.model_path.parent / Path(
                    self.model_path.stem
                    + "{}_iter{:06d}".format(self.al_iter, it)
                    + self.model_path.suffix
                )
                torch.save(self.model.state_dict(), path)
            # Moving average of the loss for early stopping
            if loss_term_ema and loss_flow_ema:
                loss_term_ema = (
                    self.ema_alpha * losses[1] + (1.0 - self.ema_alpha) * loss_term_ema
                )
                loss_flow_ema = (
                    self.ema_alpha * losses[2] + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:
                loss_term_ema = losses[1]
                loss_flow_ema = losses[2]

            # Log times
            t1_iter = time.time()
            times.update({"iter": t1_iter - t0_iter})
            times = {"time_{}{}".format(k, self.al_iter): v for k, v in times.items()}
            if self.comet and not self.no_log_times:
                self.comet.log_metrics(times, step=it)
        # Save final model
        if self.model_path:
            path = self.model_path.parent / Path(
                self.model_path.stem + "_final" + self.model_path.suffix
            )
            torch.save(self.model.state_dict(), path)
            torch.save(self.model.state_dict(), self.model_path)

        # Close comet
        if self.comet and self.al_iter == -1:
            self.comet.end()


def batch2dict(batch, env, get_uncertainties=False, query_function="Both"):
    batch = np.asarray(env.state2oracle(batch))
    t0_proxy = time.time()
    if get_uncertainties:
        if query_function == "fancy_acquisition":
            scores, proxy_vals, uncertainties = env.proxy(batch, query_function)
        else:
            proxy_vals, uncertainties = env.proxy(batch, query_function)
            scores = proxy_vals
    else:
        proxy_vals = env.proxy(batch)
        uncertainties = None
        scores = proxy_vals
    t1_proxy = time.time()
    times = {"proxy": t1_proxy - t0_proxy}
    samples = {
        "samples": batch.astype(np.int64),
        "scores": scores,
        "energies": proxy_vals,
        "uncertainties": uncertainties,
    }
    return samples, times


class RandomTrajAgent:
    def __init__(self, args, envs):
        self.mbsize = args.gflownet.mbsize  # mini-batch size
        self.envs = envs
        self.nact = args.ndim + 1
        self.model = None

    def parameters(self):
        return []

    def sample_batch(self, mbsize, all_visited):
        batch = []
        [i.reset()[0] for i in self.envs]  # reset envs
        done = [False] * mbsize
        while not all(done):
            acts = np.random.randint(0, self.nact, mbsize)  # actions (?)
            # step : list
            # - For each e in envs, if corresponding done is False
            #   - For each element i in env, and a in acts
            #     - i.step(a)
            step = [
                i.step(a)
                for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(tuple(sp))
        return []  # agent is stateful, no need to return minibatch data

    def flowmatch_loss(self, it, batch):
        return None


def make_mlp(layers_dim, act=nn.LeakyReLU(), tail=[]):
    """
    Defines an MLP with no top layer activation

    Args
    ----
    layers_dim : list
        Dimensionality of each layer

    act : Activation
        Activation function
    """
    return nn.Sequential(
        *(
            sum(
                [
                    [nn.Linear(idim, odim)] + ([act] if n < len(layers_dim) - 2 else [])
                    for n, (idim, odim) in enumerate(zip(layers_dim, layers_dim[1:]))
                ],
                [],
            )
            + tail
        )
    )


def make_opt(params, Z, args):
    """
    Set up the optimizer
    """
    params = list(params)
    if not len(params):
        return None
    if args.gflownet.opt == "adam":
        opt = torch.optim.Adam(
            params,
            args.gflownet.learning_rate,
            betas=(args.gflownet.adam_beta1, args.gflownet.adam_beta2),
        )
        if Z is not None:
            opt.add_param_group(
                {
                    "params": Z,
                    "lr": args.gflownet.learning_rate * args.gflownet.lr_z_mult,
                }
            )
    elif args.gflownet.opt == "msgd":
        opt = torch.optim.SGD(
            params, args.gflownet.learning_rate, momentum=args.gflownet.momentum
        )
    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=args.gflownet.lr.decay_period,
        gamma=args.gflownet.lr.decay_gamma,
    )
    return opt, lr_scheduler


def empirical_distribution_error(env, visited):
    """
    Computes the empirical distribution errors, as the mean L1 error and the KL
    divergence between the true density of the space and the estimated density from all
    states visited.
    """
    true_density, _, states_term = env.true_density()
    if true_density is None:
        return None, None
    true_density = tf(true_density)
    if not len(visited):
        return 1, 100
    hist = defaultdict(int)
    for s in visited:
        hist[s] += 1
    Z = sum([hist[s] for s in states_term])
    estimated_density = tf([hist[s] / Z for s in states_term])
    # L1 error
    l1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
    return l1, kl


def logq(path_list, actions_list, model, env):
    # TODO: this method is probably suboptimal, since it may repeat forward calls for
    # the same nodes.
    log_q = torch.tensor(1.0)
    for path, actions in zip(path_list, actions_list):
        path = path[::-1]
        actions = actions[::-1]
        path_obs = np.asarray([env.state2obs(state) for state in path])
        with torch.no_grad():
            # TODO: potentially mask invalid actions next_q
            logits_path = model(tf(path_obs))
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        logprobs_path = logsoftmax(logits_path)
        log_q_path = torch.tensor(0.0)
        for s, a, logprobs in zip(*[path, actions, logprobs_path]):
            log_q_path = log_q_path + logprobs[a]
        # Accumulate log prob of path
        if torch.le(log_q, 0.0):
            log_q = torch.logaddexp(log_q, log_q_path)
        else:
            log_q = log_q_path
    return log_q.item()


def main(args):
    gflownet_agent = GFlowNetAgent(args)
    gflownet_agent.train()

    # sample from the oracle, not from a proxy model
    batch, times = gflownet_agent.sample_batch(
        gflownet_agent.env, args.gflownet.n_samples, train=False
    )
    samples, times = batch2dict(batch, gflownet_agent.env, get_uncertainties=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    config = process_config(config)
    print("Config file: " + config.yaml_config)
    if config.workdir is not None:
        print("Working dir: " + config.workdir)
    print(
        "Config:\n"
        + "\n".join([f"    {k:20}: {v}" for k, v in vars(config.gflownet).items()])
    )
    if "workdir" in config and config.workdir is not None:
        if not Path(config.workdir).exists() or config.overwrite_workdir:
            Path(config.workdir).mkdir(parents=True, exist_ok=True)
            with open(config.workdir + "/config.yml", "w") as f:
                yaml.dump(
                    numpy2python(namespace2dict(config)), f, default_flow_style=False
                )
            torch.set_num_threads(1)
            main(config)
        else:
            print(f"workdir {config.workdir} already exists! - Ending run...")
    else:
        print(f"working directory not defined - Ending run...")
