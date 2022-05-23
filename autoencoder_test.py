import time
import subprocess
from multiprocessing import Process

import torch
import torchvision

import blocks
from utils import get_parameter_count


def explore(N, folder, idx):
    subprocess.run(["python", "-W", "ignore", "explore_mnist.py", "-N", N, "-o", folder, "-i", idx, "-s", "4"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":

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

    # action_w = torch.nn.Parameter(
    #     torch.nn.init.kaiming_normal(torch.empty(4*256*7*7, 512, device=DEVICE)).reshape(4, 256*7*7, 512).permute(0, 2, 1)
    # )
    # action_b = torch.nn.Parameter(
    #     torch.zeros(4, 256*7*7, device=DEVICE)
    # )

    tree = blocks.HME(gating_features=512+4, in_features=512, out_features=128, depth=5, gumbel="hard")

    decoder = torch.nn.Sequential(
        # blocks.MLP([512+4, 256*7*7]),
        torch.nn.ReLU(),
        blocks.MLP([128, 256*7*7]),
        torch.nn.ReLU(),
        blocks.Reshape([-1, 256, 7, 7]),
        # blocks.ConvTransposeBlock(512, 256, 7, 1, 0, batch_norm=BATCH_NORM),
        blocks.ConvTransposeBlock(256, 256, 4, 2, 1, batch_norm=BATCH_NORM),
        blocks.ConvTransposeBlock(256, 128, 4, 2, 1, batch_norm=BATCH_NORM),
        blocks.ConvTransposeBlock(128, 64, 4, 2, 1, batch_norm=BATCH_NORM),
        torch.nn.ConvTranspose2d(64, 3, 4, 2, 1)
    )

    encoder = encoder.to(DEVICE)
    tree = tree.to(DEVICE)
    decoder = decoder.to(DEVICE)
    print(encoder)
    print(tree)
    print(decoder)

    print(f"Encoder params: {get_parameter_count(encoder):,}")
    print(f"Tree params: {get_parameter_count(tree):,}")
    # print(f"Action module params: {get_parameter_count(action_modules):,}")
    print(f"Decoder params: {get_parameter_count(decoder):,}")
    optimizer = torch.optim.Adam(lr=LR,
                                 params=[
                                            {"params": encoder.parameters()},
                                            {"params": tree.parameters()},
                                            {"params": decoder.parameters()},
                                            # {"params": [action_w, action_b]},
                                        ],
                                 amsgrad=True)
    criterion = torch.nn.MSELoss()

    best_loss = 1e100
    for e in range(EPOCH):
        start = time.time()
        # collect 100,000 samples
        procs = []
        for j in range(NUM_WORKERS):
            p = Process(target=explore, args=[str(N_PER_WORKER), "data/mnistnpuzzle/", str(j)])
            p.start()
            procs.append(p)
        for j in range(NUM_WORKERS):
            procs[j].join()
        end = time.time()
        collect_elapsed = end - start

        X = torch.cat([torch.load(f"data/mnistnpuzzle/state{j}.pt") for j in range(NUM_WORKERS)], dim=0)
        A = torch.cat([torch.load(f"data/mnistnpuzzle/action{j}.pt") for j in range(NUM_WORKERS)], dim=0)
        E = torch.cat([torch.load(f"data/mnistnpuzzle/effect{j}.pt") for j in range(NUM_WORKERS)], dim=0)
        assert X.shape[0] == N

        R = torch.randperm(N)
        avg_loss = 0.0
        start = time.time()
        for i in range(LOOP_PER_EPOCH):
            index = R[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            x_i = X[index].to(DEVICE) / 255.0
            a_i = A[index].to(DEVICE).float()
            e_i = E[index].to(DEVICE) / 255.0
            # encoding
            h_i = encoder(x_i)

            # gating
            # gated_w = (a_i @ action_w.flatten(1, -1)).reshape(a_i.shape[0], action_w.shape[1], action_w.shape[2])
            # gated_b = a_i @ action_b
            # z_i = (h_i.unsqueeze(1) @ gated_w).squeeze() + gated_b
            g_i = torch.cat([h_i, a_i], dim=-1)
            z_i = tree(g_i, h_i)

            # decoding
            e_bar = decoder(z_i)
            loss = criterion(e_bar, e_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        end = time.time()
        train_elapsed = end - start
        avg_loss /= i+1
        print(f"epoch: {e+1}, loss: {avg_loss:.7f}, collect time: {collect_elapsed:.2f}, train time: {train_elapsed:.2f}")
        x_all = torch.cat([e_i[:10].cpu(), (e_bar[:10].cpu().detach().clamp(-1., 1.))], dim=0)
        torchvision.utils.save_image(x_all, f"save/npuzzle_ae_tree/epoch{e+1}.png", nrow=10, normalize=True, pad_value=1.0)
        torch.save(encoder.cpu().eval(), "save/npuzzle_ae_tree/encoder_last.bin")
        torch.save(tree.cpu().eval(), "save/npuzzle_ae_tree/tree_last.bin")
        torch.save(decoder.cpu().eval(), "save/npuzzle_ae_tree/decoder_last.bin")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder, "save/npuzzle_ae_head/encoder_best.bin")
            torch.save(tree, "save/npuzzle_ae_tree/tree_best.bin")
            torch.save(decoder, "save/npuzzle_ae_head/decoder_best.bin")
        encoder.to(DEVICE).train()
        tree.to(DEVICE).train()
        decoder.to(DEVICE).train()
