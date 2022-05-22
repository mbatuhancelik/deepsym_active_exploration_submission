import subprocess
from multiprocessing import Process

import torch
import torchvision

import blocks
from utils import get_parameter_count


def explore(N, folder, idx):
    subprocess.run(["python", "-W", "ignore", "explore_crafter.py", "-N", N, "-o", folder, "-i", idx],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 9600
BATCH_SIZE = 32
LOOP_PER_EPOCH = N // BATCH_SIZE
EPOCH = 500
BATCH_NORM = False
LR = 0.001
NUM_WORKERS = 12
N_PER_WORKER = N // NUM_WORKERS

encoder = torch.nn.Sequential(
    blocks.ConvBlock(3, 64, 4, 2, 1, batch_norm=BATCH_NORM),
    blocks.ConvBlock(64, 128, 4, 2, 1, batch_norm=BATCH_NORM),
    blocks.ConvBlock(128, 256, 4, 2, 1, batch_norm=BATCH_NORM),
    blocks.ConvBlock(256, 512, 4, 2, 1, batch_norm=BATCH_NORM),
    blocks.Avg([2, 3])
)

# action_modules = torch.nn.ModuleList([torch.nn.Linear(512, 256*4*4) for _ in range(17)])
decoder = torch.nn.Sequential(
    blocks.MLP([512+17, 256*4*4]),
    blocks.Reshape([-1, 256, 4, 4]),
    # blocks.ConvTransposeBlock(512, 256, 4, 1, 0, batch_norm=BATCH_NORM),
    blocks.ConvTransposeBlock(256, 256, 4, 2, 1, batch_norm=BATCH_NORM),
    blocks.ConvTransposeBlock(256, 128, 4, 2, 1, batch_norm=BATCH_NORM),
    blocks.ConvTransposeBlock(128, 64, 4, 2, 1, batch_norm=BATCH_NORM),
    torch.nn.ConvTranspose2d(64, 3, 4, 2, 1)
)

encoder = encoder.to(DEVICE)
# action_modules = action_modules.to(DEVICE)
decoder = decoder.to(DEVICE)
print(encoder)
print(decoder)
print(f"Encoder params: {get_parameter_count(encoder):,}")
# print(f"Action module params: {get_parameter_count(action_modules):,}")
print(f"Decoder params: {get_parameter_count(decoder):,}")
optimizer = torch.optim.Adam(lr=LR,
                             params=[
                                        {"params": encoder.parameters()},
                                        {"params": decoder.parameters()}
                                    ],
                             amsgrad=True)
criterion = torch.nn.MSELoss()

best_loss = 1e100
for e in range(EPOCH):
    # collect 100,000 samples
    procs = []
    for j in range(NUM_WORKERS):
        p = Process(target=explore, args=[str(N_PER_WORKER), "data/crafter_data/", str(j)])
        p.start()
        procs.append(p)
    for j in range(NUM_WORKERS):
        procs[j].join()

    X = torch.cat([torch.load(f"data/crafter_data/state{j}.pt") for j in range(NUM_WORKERS)], dim=0)
    A = torch.cat([torch.load(f"data/crafter_data/action{j}.pt") for j in range(NUM_WORKERS)], dim=0)
    E = torch.cat([torch.load(f"data/crafter_data/effect{j}.pt") for j in range(NUM_WORKERS)], dim=0)
    assert X.shape[0] == N

    R = torch.randperm(N)
    avg_loss = 0.0
    for i in range(LOOP_PER_EPOCH):
        index = R[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        x_i = X[index].to(DEVICE) / 255.0
        a_i = A[index].to(DEVICE)
        e_i = E[index].to(DEVICE) / 255.0
        h_i = encoder(x_i)
        z_i = torch.cat([h_i, a_i], dim=-1)
        e_bar = decoder(z_i)
        loss = criterion(e_bar, e_i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss /= i+1
    print(f"epoch: {e+1}, loss: {avg_loss:.7f}")
    x_all = torch.cat([e_i[:10].cpu(), (e_bar[:10].cpu().detach().clamp(-1., 1.))], dim=0)
    torchvision.utils.save_image(x_all, f"save/crafter_ae/epoch{e+1}.png", nrow=10, normalize=True)
    torch.save(encoder, "save/crafter_ae/encoder_last.bin")
    torch.save(decoder, "save/crafter_ae/decoder_last.bin")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(encoder, "save/crafter_ae/encoder_best.bin")
        torch.save(decoder, "save/crafter_ae/decoder_best.bin")
