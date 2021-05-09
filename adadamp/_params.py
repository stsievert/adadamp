from copy import deepcopy
lr = 0.0173887882
base = {"lr": lr, "weight_decay": 3e-4, "momentum": 0.9}
params = {
    "geodamp": {"dampingfactor": 5, "dampingdelay": 6, "max_batch_size": 4096, "initial_batch_size": 64, **base},
    "radadamp": { "initial_batch_size": 64, "max_batch_size": 2048, **base},
    "adagrad": {"lr": 0.0056328758, "initial_batch_size": 256},
}
params["geodamplr"] = deepcopy(params["geodamp"])
params["geodamplr"]["max_batch_size"] = params["geodamplr"]["initial_batch_size"]

from pprint import pprint
pprint(params)

#  with open("hyperparams.json", "r"):
    #  json.dump(params, f)
