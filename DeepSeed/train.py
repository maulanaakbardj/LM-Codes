import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import FusedAdam
from deepspeed.utils.logging import logger

# Inisialisasi model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = Model()

# Inisialisasi dataset dan data loader
data = torch.randn(100, 10)
target = torch.randint(2, size=(100,))

dataset = torch.utils.data.TensorDataset(data, target)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Inisialisasi optimizer
optimizer = FusedAdam(model.parameters(), lr=1e-3)

# Inisialisasi DeepSpeed engine
engine, optimizer, _, _ = deepspeed.initialize(optimizer=optimizer)

# Training loop
for epoch in range(10):
    for batch in data_loader:
        data, target = batch
        data = data.to(engine.device)
        target = target.to(engine.device)

        engine.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        engine.backward(loss)
        engine.step()

    logger.info(f'Epoch {epoch}: loss={loss.item()}')
