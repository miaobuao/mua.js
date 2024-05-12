import torch as th
from torch import nn
from torch.optim import SGD


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 2, 5, stride=2)
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x: th.Tensor = self.conv1(x)
        x = x.tanh()
        x = self.conv2(x)
        x = x.flatten()
        x = self.linear(x)
        x = x.relu()
        return x


# class MyModel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.linear1 = nn.Linear(28 * 28, 32)
#         self.linear2 = nn.Linear(32, 10)

#     def forward(self, x):
#         x = x.flatten()
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x


if __name__ == "__main__":

    from pathlib import Path
    from random import shuffle
    import os
    import time
    from pydantic import BaseModel
    from PIL import Image
    import numpy as np

    class DataItem(BaseModel):
        label: str
        path: Path

    def load_dataset(path: Path):
        for label in os.listdir(dataset_path):
            for file in path.joinpath(label).iterdir():
                yield DataItem(label=label, path=file)

    cwd = Path(__file__).parent
    dataset_path = cwd.joinpath("../demo/MNIST")
    datasets = list(load_dataset(dataset_path))
    shuffle(datasets)

    model = MyModel()
    loss = nn.CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=1e-3)
    forward_time, backward_time = [], []
    for d in datasets[:2000]:
        st = time.time()
        opt.zero_grad()
        img = Image.open(d.path)
        x = np.array(img)
        x = th.from_numpy(x).float().unsqueeze(0)
        y = th.tensor([int(d.label)]).long()
        z = model(x).unsqueeze(0)
        ed = time.time()
        forward_time.append(ed - st)
        st = time.time()
        l = loss(z, y)
        l.backward()
        opt.step()
        ed = time.time()
        backward_time.append(ed - st)
    print("forward: ", sum(forward_time) / len(forward_time) * 1000)
    print("backward: ", sum(backward_time) / len(backward_time) * 1000)
     