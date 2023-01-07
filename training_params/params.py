from dataclasses import dataclass


@dataclass
class TrainingParams:
    device: int = 0
    nepochs: int = 400
    nsims: int = 1


def selected_device(id: int):
    device = 'cuda' if id == 0 else 'cpu'
    return device


training_params = TrainingParams(0, 400, 120)
