import torch
target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 1.5)  # A prediction (logit)
pos_weight = torch.ones([64])  # All weights are equal to 1
weight = torch.ones([10, 64]) * 2
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
loss = criterion(output, target)  # -log(sigmoid(1.5))
loss = loss * weight
loss = torch.mean(loss)
print(loss)