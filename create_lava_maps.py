import lava
import pickle
import numpy as np

from otdd.pytorch.datasets import dataset_from_numpy

reward_min = 50
reward_max = 100
point_labels_file = "../datacollect/experiments/reforestree/simulations/300x300/env/point_labels.pkl"
data_file = (
    "../datacollect/experiments/reforestree/simulations/300x300/env/data.pkl"
)
save_file = "../datacollect/experiments/reforestree/simulations/300x300/env/reward_maps/lava.pkl"

with open(point_labels_file, "rb") as f:
    point_labels = pickle.load(f)

with open(data_file, "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

train_dataset = dataset_from_numpy(X_train.to_numpy(), y_train.to_numpy())
test_dataset = dataset_from_numpy(X_test.to_numpy(), y_test.to_numpy())
training_size = 1000
dual_sol = lava.get_OT_dual_sol(
    feature_extractor="euclidean",
    trainloader=train_dataset,
    testloader=test_dataset,
    training_size=training_size,
)
vals = np.array(lava.values(dual_sol=dual_sol, training_size=training_size))
vals = np.interp(vals, (vals.min(), vals.max()), (reward_min, reward_max))
reward_map = {label: vals[i] for i, label in enumerate(point_labels)}

print(vals)
print(len(vals) == len(point_labels))

with open(save_file, "wb") as f:
    pickle.dump(reward_map, f)
