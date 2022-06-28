import torch

from models import DeepSymv3
import blocks
import utils
import environment
from dataset import preprocess


device = "cuda" if torch.cuda.is_available() else "cpu"

STATE_BITS = 8
ACTION_BITS = 16
BN = True
EFF_MU = torch.tensor([-0.0003,  0.0002,  0.0004])
EFF_STD = torch.tensor([0.0687, 0.0937, 0.0164])


encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.MLP([512, STATE_BITS]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=STATE_BITS+ACTION_BITS, nhead=1, batch_first=True)
decoder_att = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
decoder = blocks.MLP([STATE_BITS+ACTION_BITS, 256, 256, 256, 3], batch_norm=BN)

encoder.to(device)
decoder.to(device)
decoder_att.to(device)

model = DeepSymv3(encoder=encoder, decoder=decoder, decoder_att=decoder_att,
                  device=device, lr=0.0001, path="save/3x5_t4_4b_8/", coeff=1.0)
model.load("_best")
model.eval_mode()

model.print_model()
for name in model.module_names:
    print(f"{name} params={utils.get_parameter_count(getattr(model, name)):,}")
    for param in getattr(model, name).parameters():
        param.requires_grad = False

valid_objects = {i: True for i in range(4, 13)}
env = environment.BlocksWorld_v2(gui=1, min_objects=5, max_objects=9)
env._step(10000)

while True:
    _, depth_img, seg = env.state()
    depth_img = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()))*255
    depth_img = torch.tensor(depth_img).unsqueeze(0)
    seg = torch.tensor(seg)
    padded, pad_mask = preprocess(depth_img, seg, valid_objects, env.num_objects)
    n = padded.shape[0]
    action = input()
    if action == "reset":
        env.reset_objects()
        continue
    elif action[:4] == "wait":
        steps = int(action[4:].rstrip())
        env._step(steps)
        continue
    elif action == "exit":
        del env
        break
    else:
        action = torch.tensor([int(a) for a in action], dtype=torch.long)
        sample = {
            "state": padded.unsqueeze(0),
            "action": action.unsqueeze(0),
            "pad_mask": pad_mask.unsqueeze(0)
        }
        pred_z, pred_e = model.forward(sample, eval_mode=True)
        pred_e = pred_e*EFF_STD + EFF_MU
        effect = env.step(action[:2].tolist(), action[2:].tolist())
        print("Predicted // Actual")
        print("------------------")
        for p_e, a_e in zip(pred_e[0], effect):
            print(f"{p_e[0]:.3f}, {p_e[1]:.3f}, {p_e[2]:.3f} // {a_e[0]:.3f}, {a_e[1]:.3f}, {a_e[2]:.3f}")
