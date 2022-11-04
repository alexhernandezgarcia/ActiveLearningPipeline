"""
Logger utils, to be developed.
"""
import wandb
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    """
    Utils functions to compute and handle the statistics (saving them or send to
    comet).  Incorporates the previous function "getModelState", ...  Like
    FormatHandler, it can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(self, config):
        self.config = config
        date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
        run_name = "al{}_{}_proxy{}_oracle{}_gfn{}_{}".format(
            config.al.mode,
            config.env.main.upper(),
            config.proxy.model.upper(),
            config.oracle.main.upper(),
            config.gflownet.policy_model.upper(),
            date_time,
        )
        self.run = wandb.init(
            config=config, project="ActiveLearningPipeline", name=run_name
        )
        self.context = "0"

    def set_context(self, context):
        self.context = context

    def log_metric(self, key, value, use_context=True):
        if use_context:
            key = self.context + "/" + key
        wandb.log({key: value})

    def log_histogram(self, key, value, use_context=True):
        # need this condition for when we are training gfn without active learning and context = ""
        # we can't make use_context=False because then when the same gfn is used with AL, context won't be recorded (undesirable)
        if use_context:
            key = self.context + "/" + key
        fig = plt.figure()
        plt.hist(value)
        plt.title(key)
        plt.ylabel("Frequency")
        plt.xlabel(key)
        fig = wandb.Image(fig)
        wandb.log({key: fig})

    def finish(self):
        wandb.finish()
