import torch
import torch.nn.functional as F
import json
import math
from model import TinyGPT
from dataset import TinyDataset
from main import set_seed
from telemetry import TelemetryLogger

def run_training_segment(start_step, end_step, checkpoint_path_to_load=None, log_file="audit.jsonl"):
    set_seed(99)
    dataset = TinyDataset()
    model = TinyGPT(vocab_size=dataset.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    logger = TelemetryLogger(filepath=log_file)

    if checkpoint_path_to_load:
        checkpoint = torch.load(checkpoint_path_to_load, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f" ~> Auditor loaded checkpoint from step {start_step}")

    x, y = dataset.get_batch()

    for step in range(start_step, end_step):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_step(step, loss.item(), model)

        if not checkpoint_path_to_load and step == 4:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, "mid_checkpoint.pt")
            print(f" ~> Prover saved checkpoint at step 5")
            
if __name__ == "__main__":
    print("\n SEGMENTED AUDIT VERIFICATION ")

    print("\n[Running Prover: Steps 0 to 10]")
    run_training_segment(start_step = 0, end_step = 10, log_file="prover_log.jsonl")

    print("\n[Running Auditor: Steps 5 to 10 with checkpoint]")
    run_training_segment(start_step = 5, end_step = 10, checkpoint_path_to_load="mid_checkpoint.pt", log_file="auditor_log.jsonl")

    print("\n[Verifying Telemetry Trajectories]")
    with open("prover_log.jsonl", "r") as f:
        prover_logs = [json.loads(line) for line in f.readlines()]
    with open("auditor_log.jsonl", "r") as f:
        auditor_logs = [json.loads(line) for line in f.readlines()]

    prover_segment = prover_logs[5:10]

    if len(prover_segment) != 5 or len(auditor_logs) != 5:
        print(f"Log length mismatch — prover_segment: {len(prover_segment)}, auditor: {len(auditor_logs)}")
        print("Cannot verify. Check for crashes or early exits in training.")
    else:
        match = True

        for p, a in zip(prover_segment, auditor_logs):
            step_match = p['step'] == a['step']
            loss_match = math.isclose(p['loss'], a['loss'], rel_tol=1e-6)
            step_ok = step_match and loss_match
            if not step_ok:
                match = False
            status = "ok" if step_ok else "error"
            print(f"Step {p['step']} | Prover Loss: {p['loss']:.6f} | Auditor Loss: {a['loss']:.6f} {status}")

        if match:
            print("\n (❁ ´◡`❁) \nAUDIT PASSED: Segment replay is bitwise deterministic.")
        else:
            print("\n (╯°□°）╯︵ ┻━┻ \nAUDIT FAILED: Trajectories diverged.")

#Reproducibility test for tinyGPT without Segment Verification test
#uncomment this block and comment the others to not have segment verification
'''
import torch
#for linear model: from model import TinyModel
from model import TinyGPT
from dataset import TinyDataset
import torch.nn.functional as F
from main import set_seed

def train_once():
    set_seed(99)

    dataset = TinyDataset()
    #for linear model: model = TinyModel(vocab_size=dataset.vocab_size)
    model = TinyGPT(vocab_size=dataset.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    x, y = dataset.get_batch()

    for step in range(5):
        logits = model(x)
        #for linear model: loss = F.cross_entropy(logits, y[0])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses

if __name__ == "__main__":
    print("running reproducability test")

    model1, losses1 = train_once()

    model2, losses2 = train_once()

    losses_match = (losses1 ==losses2)
    print(f"Loss curves identical: {losses_match}")
    
    params_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model1.parameters(), model2.parameters())
    )
    print (f"Bitwise parameter match: {params_match}")

    if losses_match and params_match:
        print("\nSuccess!: Full deterministic gradient flow verified.")
    else:
        print ("\nFailure: Entropy led to non-deterministic behavior")

'''