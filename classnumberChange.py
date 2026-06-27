import torch

model1 = torch.load('./detr-r50-e632da11.pth')

num_class = 4
model1["model"]["class_embed.weight"].resize_(num_class + 1, 256)
model1["model"]["class_embed.bias"].resize_(num_class + 1)
torch.save(model1, "detr-r50_%d.pth" % num_class)