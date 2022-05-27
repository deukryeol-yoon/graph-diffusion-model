from utils import *
import torch
import dgl
from evaluation.evaluator import Evaluator

if __name__ == "__main__":
    grids = make_grid_graphs()
    lobsters = make_lobster_graphs()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    grids = [dgl.DGLGraph(g).to(device) for g in grids] # Convert graphs to DGL from NetworkX
    lobsters = [dgl.DGLGraph(g).to(device) for g in lobsters] # Convert graphs to DGL from NetworkX
    
    evaluator = Evaluator(device=device)
    eval_result = evaluator.evaluate_all(generated_dataset=grids, reference_dataset=lobsters) # KID eval is deleted because of the issue on the tensorflow version.
    print(eval_result)