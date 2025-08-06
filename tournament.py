# tournament.py
import torch
import os
from app import evaluate_model, get_data_loaders

class EvolutionTournament:
    def __init__(self):
        self.train_loader, self.test_loader = get_data_loaders()
        self.agents = self.load_agents()

    def load_agents(self):
        agents = {}
        os.makedirs("agents", exist_ok=True)
        for f in os.listdir("checkpoints"):
            if f.endswith(".pth"):
                path = f"checkpoints/{f}"
                # Assume resnet-like arch
                model = torch.hub.load("pytorch/vision", "resnet18", weights=None, num_classes=10)
                try:
                    model.load_state_dict(torch.load(path, map_location="cpu"))
                    model.eval()
                    agents[f] = model
                except:
                    pass
        return agents

    def run_tournament(self):
        results = {}
        for name, model in self.agents.items():
            acc = evaluate_model(model, self.test_loader)
            results[name] = acc
        # Sort
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_results[0]
        return sorted_results, winner