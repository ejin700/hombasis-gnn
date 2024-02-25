import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error



def train(model_name, use_counts, model, device, loader, optimizer):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """

    loss_fn = torch.nn.L1Loss()

    curve = list()
    model.train()
    # for step, batch in enumerate(tqdm(loader, desc="Training iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)  # batch.cuda() if torch.cuda.is_available() else batch

        optimizer.zero_grad()
        
        if model_name == "GINGraphReg" or model_name == "GCNGraphReg" or model_name == "GATGraphReg":
            pred = model(x=batch.x, edge_index=batch.edge_index, counts=batch.counts, use_counts=use_counts, batch=batch.batch)
        else:
            print('model not supported')
            
        targets = batch.y.to(torch.float32).view(pred.shape)

        loss = loss_fn(pred, targets)

        loss.backward()
        optimizer.step()
        curve.append(loss.detach().cpu().item())

    return curve


def eval(model_name, use_counts, model, device, loader):
    """
        Evaluates a model over all the batches of a data loader.
    """
        
    loss_fn = torch.nn.L1Loss()

    model.eval()
    y_true = []
    y_pred = []
    losses = []
    # for step, batch in enumerate(tqdm(loader, desc="Eval iteration")):
    for step, batch in enumerate(loader):
        # Cast features to double precision if that is used
        if torch.get_default_dtype() == torch.float64:
            for dim in range(batch.dimension + 1):
                batch.cochains[dim].x = batch.cochains[dim].x.double()
                assert batch.cochains[dim].x.dtype == torch.float64, batch.cochains[dim].x.dtype

        batch = batch.to(device)
        with torch.no_grad():
            if model_name == "GINGraphReg" or model_name == "GCNGraphReg" or model_name == "GATGraphReg":
                pred = model(x=batch.x, edge_index=batch.edge_index, counts=batch.counts, use_counts=use_counts, batch=batch.batch)
            else:
                print('model not supported')
            
            targets = batch.y.to(torch.float32).view(pred.shape)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            loss = loss_fn(pred, targets)
            losses.append(loss.detach().cpu().item())

        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy() if len(y_true) > 0 else None
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    assert y_true is not None
    assert y_pred is not None
    
    mae = mean_absolute_error(y_true, y_pred)
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
    
    return mae, mean_loss


def parse_args():
    parser = argparse.ArgumentParser(description='ZINC experiment')
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-project", "--project")
    parser.add_argument("-group","--group", help="group name on wandb", required=True)
    parser.add_argument("-seed", "--seed", help="seed", required=True)

    args, unparsed = parser.parse_known_args()
    
    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    conf['project'] = args.project
    conf['group'] = args.group
    conf['seed'] = args.seed
    
    return(conf)
