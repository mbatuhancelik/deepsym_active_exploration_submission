import torch

from models import DeepSymv3
import blocks
import utils
import environment


device = "cuda" if torch.cuda.is_available() else "cpu"

STATE_BITS = 8
ACTION_BITS = 12
BN = True
NUM_INTERACTION = 10000

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
env = environment.BlocksWorld_v2(gui=0, min_objects=1, max_objects=3)

eye = torch.eye(6)
i = 0
env_it = 0
symbol_forward_map = {}
while i < NUM_INTERACTION:
    state_tuple = env.state()
    padded, pad_mask = utils.state_to_tensor(state_tuple, valid_objects=valid_objects, num_objects=env.num_objects, is_old=False)
    z_before = model.encode(padded.unsqueeze(0), pad_mask.unsqueeze(0))[0]
    from_idx, to_idx = env.sample_random_action()
    action_vector = torch.cat([eye[from_idx], eye[to_idx]], dim=-1).unsqueeze(0)
    env.step(from_idx, to_idx)

    state_tuple = env.state()
    padded, pad_mask = utils.state_to_tensor(state_tuple, valid_objects=valid_objects, num_objects=env.num_objects, is_old=False)
    z_after = model.encode(padded.unsqueeze(0), pad_mask.unsqueeze(0))[0]
    z_i = sorted(utils.binary_tensor_to_str(z_before))
    a = utils.binary_tensor_to_str(action_vector)
    z_f = sorted(utils.binary_tensor_to_str(z_after))
    precond = (tuple(z_i), tuple(a))
    effect = tuple(z_f)
    if precond in symbol_forward_map:
        if effect in symbol_forward_map[precond]:
            symbol_forward_map[precond][effect] += 1
        else:
            symbol_forward_map[precond][effect] = 1
    else:
        symbol_forward_map[precond] = {effect: 1}

    env_it += 1
    if env_it == 20:
        env_it = 0
        env.reset_objects()
        for precond in symbol_forward_map:
            for effect in symbol_forward_map[precond]:
                print(f"{precond}->{effect}: {symbol_forward_map[precond][effect]}")
        continue
