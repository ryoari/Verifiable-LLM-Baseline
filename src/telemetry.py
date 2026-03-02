import json
import torch
import os

class TelemetryLogger:
    def __init__(self, filepath="audit_log.jsonl"):
        self.filepath = filepath

        open(self.filepath, 'w').close()

    def log_step(self, step, loss, model):
        grad_norm = 0.0
        param_norm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
            param_norm += p.data.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5
        param_norm = param_norm ** 0.5

        record ={
            "step": step,
            "loss": round(loss, 8),
            "grad_norm": round(grad_norm, 8),
            "param_norm": round(param_norm, 8)
        }

        with open(self.filepath, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        return record