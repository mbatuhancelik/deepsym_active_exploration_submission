import torch

from models import DeepSymv3
import blocks
import utils
import environment
from dataset import preprocess


device = utils.get_device()

STATE_BITS = 8
ACTION_BITS = 12
BN = True
EFF_MU = torch.tensor([8.0498e-03,  2.4965e-04,  4.6637e-04])
EFF_STD = torch.tensor([0.1055, 0.1474, 0.0296])


encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.MLP([512, STATE_BITS]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

projector = torch.nn.Linear(STATE_BITS+ACTION_BITS, 128)
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
decoder_att = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
decoder = blocks.MLP([128, 256, 256, 256, 3], batch_norm=BN)

encoder.to(device)
decoder.to(device)
projector.to(device)
decoder_att.to(device)

model = DeepSymv3(encoder=encoder, decoder=decoder, decoder_att=decoder_att, projector=projector,
                  device=device, lr=0.0001, path="save/v4_varying_new/", coeff=1.0)
model.load("_best")
model.eval_mode()

model.print_model()
for name in model.module_names:
    print(f"{name} params={utils.get_parameter_count(getattr(model, name)):,}")
    for param in getattr(model, name).parameters():
        param.requires_grad = False

valid_objects = {i: True for i in range(4, 7)}
env = environment.BlocksWorld_v2(gui=1, min_objects=1, max_objects=3)
env._step(10000)
eff_arrow_ids = []
while True:
    while len(eff_arrow_ids) > 0:
        arrow_id = eff_arrow_ids.pop()
        env._p.removeBody(arrow_id)
    _, depth_img, seg = env.state()
    depth_img = utils.normalize_depth_img(depth_img)
    depth_img = torch.tensor(depth_img).unsqueeze(0)
    seg = torch.tensor(seg)
    padded, pad_mask = preprocess(depth_img, seg, valid_objects, env.num_objects, aug=False, old=False)
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
        eye = torch.eye(6)
        sample = {
            "state": padded.unsqueeze(0),
            "action": torch.cat([eye[action[0]], eye[action[1]]], dim=-1)
            .reshape(1, 1, 12).repeat(1, padded.shape[0], 1),
            "pad_mask": pad_mask.unsqueeze(0)
        }
        pred_z, pred_e = model.forward(sample, eval_mode=True)
        pred_e = pred_e*EFF_STD + EFF_MU
        poses = env.state_obj_poses()
        for o_i, e_i in zip(poses, pred_e[0]):
            eff_arrow_ids.append(utils.create_arrow(env._p, o_i[:3], o_i[:3]+e_i.numpy()))
        effect = env.step(action[0].item(), action[1].item())
        print("Predicted // Actual")
        print("------------------")
        for p_e, a_e, z in zip(pred_e[0], effect, pred_z[0]):
            print(f"{p_e[0]:.3f}, {p_e[1]:.3f}, {p_e[2]:.3f} // {a_e[0]:.3f}, {a_e[1]:.3f}, {a_e[2]:.3f}"
                  f"-- {z[0]}, {z[1]}, {z[2]}, {z[3]}, {z[4]}, {z[5]}, {z[6]}, {z[7]}")
