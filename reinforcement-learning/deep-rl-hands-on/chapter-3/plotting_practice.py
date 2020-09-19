
import math
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

funcs = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan}


for angle in range(-360, 360):
  angle_rad = angle * math.pi/180
  for name, func in funcs.items():
    val = func(angle_rad)
    writer.add_scalar(name, val, angle) # plot_name, y, x

writer.close()

# to see the results run:
# tensorboard --logdir runs --host localhost