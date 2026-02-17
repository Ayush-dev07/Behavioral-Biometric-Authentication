import torch
import numpy as np

def enroll(encoder, sequences):
    encoder.eval()

    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            seq = torch.tensor(seq).unsqueeze(0)
            emb = encoder(seq)
            embeddings.append(emb.squeeze(0))

    template = torch.stack(embeddings).mean(dim=0)
    return template
