import wandb

sweep_config = {
  "name" : "hyperparam_search",
  "method" : "bayes",
  "metric": {
    "name": "val_loss",
    "goal": "minimize"
  },
  "parameters" : {
    "optimizer": {
        "values": ["adam", "sgd"]
    },
    "learning_rate": {
      "values": [1e-4, 1e-3]
    },
    "epochs": {
        "values": [1, 20, 30]
    }
  }
}