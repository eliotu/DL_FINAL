import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import sys

from Stock_Mixer_model.src.evaluator import evaluate
sys.path.append("../../")

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    ground_truth = ground_truth.t()
    mask = mask.t()
    base_price = base_price.t()

    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float64).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


def train(
    model,
    epochs: int,
    stock_num: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    optimizer,
    alpha: float,
):
    
    history = {
        'train_total_loss': [], 
        'train_reg_loss': [], 
        'train_rank_loss': [],
        
        'val_total_loss': [], 
        'val_reg_loss': [], 
        'val_rank_loss': [],
        
        'test_total_loss': [],
        'test_reg_loss': [],
        'test_rank_loss': [],
        
        'valid_mse': [],
        'valid_ic': [],
        'valid_ric': [],
        'valid_prec10': [],
        'valid_sr': [],
        
        'test_mse': [],
        'test_ic': [],
        'test_ric': [],
        'test_prec10': [],
        'test_sr': []
    }
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}:')
        train_metrics = train_model(model, train_loader, optimizer, alpha)
        print(f"Training Loss:   {train_metrics['total_loss']:.4e}, "
                f"Regularization Loss: {train_metrics['reg_loss']:.4e}, "
                f"Ranking Loss: {train_metrics['rank_loss']:.4e}")
        
        val_metrics, cur_valid_perf = validate_model(model, valid_loader, stock_num=stock_num, alpha=alpha)
        print(f"Validation Loss: {val_metrics['total_loss']:.4e}, "
                f"Regularization Loss: {val_metrics['reg_loss']:.4e}, "
                f"Ranking Loss: {val_metrics['rank_loss']:.4e}")
        
        test_metrics, cur_test_perf = test_model(model, test_loader, stock_num=stock_num, alpha=alpha)
        print(f"Test Loss:       {test_metrics['total_loss']:.4e}, "
                f"Regularization Loss: {test_metrics['reg_loss']:.4e}, "
                f"Ranking Loss: {test_metrics['rank_loss']:.4e}")
        
        print()

        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_reg_loss'].append(train_metrics['reg_loss'])
        history['train_rank_loss'].append(train_metrics['rank_loss'])
        
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_reg_loss'].append(val_metrics['reg_loss'])
        history['val_rank_loss'].append(val_metrics['rank_loss'])
        
        history['test_total_loss'].append(test_metrics['total_loss'])
        history['test_reg_loss'].append(test_metrics['reg_loss'])
        history['test_rank_loss'].append(test_metrics['rank_loss'])
        
        history['valid_mse'].append(cur_valid_perf['mse'])
        history['valid_ic'].append(cur_valid_perf['IC'])
        history['valid_ric'].append(cur_valid_perf['RIC'])
        history['valid_prec10'].append(cur_valid_perf['prec_10'])
        history['valid_sr'].append(cur_valid_perf['sharpe5'])
        
        history['test_mse'].append(cur_test_perf['mse'])
        history['test_ic'].append(cur_test_perf['IC'])
        history['test_ric'].append(cur_test_perf['RIC'])
        history['test_prec10'].append(cur_test_perf['prec_10'])
        history['test_sr'].append(cur_test_perf['sharpe5'])
        
    return history

def train_model(
    model,
    train_loader: DataLoader,
    optimizer,
    alpha: float,
    desc: str = "Training"
):
    model.train()
    train_metrics = {
        'total_loss': 0.0,
        'reg_loss': 0.0,
        'rank_loss': 0.0,
    }

    for batch in tqdm(train_loader, desc=desc):
        stock_data, cholesky_vectors, mask, base_price, ground_truth = batch
        
        optimizer.zero_grad()
        prediction = model(stock_data, cholesky_vectors)
        
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(
            prediction, ground_truth, base_price, mask, prediction.size(0), alpha
        )
        
        cur_loss.backward()
        optimizer.step()
        
        train_metrics['total_loss'] += cur_loss.item()
        train_metrics['reg_loss'] += cur_reg_loss.item()
        train_metrics['rank_loss'] += cur_rank_loss.item() 
    
    for key in train_metrics:
        train_metrics[key] /= (len(train_loader) + 1)

    return train_metrics

def validate_model(
    model,
    valid_loader: DataLoader,
    stock_num: int,
    alpha: float,
    desc: str = "Validation"
):
    model.eval()
    
    val_metrics = {
        'total_loss': 0.0,
        'reg_loss': 0.0,
        'rank_loss': 0.0,
    }
    cur_valid_pred = np.zeros([stock_num, len(valid_loader)], dtype=float)
    cur_valid_gt = np.zeros([stock_num, len(valid_loader)], dtype=float)
    cur_valid_mask = np.zeros([stock_num, len(valid_loader)], dtype=float)
    with torch.no_grad():
        i = 0
        for batch in tqdm(valid_loader, desc=desc):
            stock_data, cholesky_vectors, mask, base_price, ground_truth = batch
            
            prediction = model(stock_data, cholesky_vectors)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(
                prediction, ground_truth, base_price, mask, prediction.size(0), alpha
            )
            
            val_metrics['total_loss'] += cur_loss.item()
            val_metrics['reg_loss'] += cur_reg_loss.item()
            val_metrics['rank_loss'] += cur_rank_loss.item()
            
            cur_valid_pred[:, i] = cur_rr[:, 0]
            cur_valid_gt[:, i] = ground_truth[:, 0]
            cur_valid_mask[:, i] = mask[:, 0]
            i += 1
            
        for key in val_metrics:
            val_metrics[key] /= (len(valid_loader) + 1)

        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return val_metrics, cur_valid_perf

def test_model(
    model,
    test_loader: DataLoader,
    stock_num: int,
    alpha: float,
    desc: str = "Testing",
):
    model.eval()
    test_metrics = {
        'total_loss': 0.0,
        'reg_loss': 0.0,
        'rank_loss': 0.0,
    }
    cur_test_pred = np.zeros([stock_num, len(test_loader)], dtype=float)
    cur_test_gt = np.zeros([stock_num, len(test_loader)], dtype=float)
    cur_test_mask = np.zeros([stock_num, len(test_loader)], dtype=float)
    with torch.no_grad():
        i = 0
        for batch in tqdm(test_loader, desc=desc):
            stock_data, cholesky_vectors, mask, base_price, ground_truth = batch
            prediction = model(stock_data, cholesky_vectors)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(
                prediction, ground_truth, base_price, mask, prediction.size(0), alpha
            )
            
            test_metrics['total_loss'] += cur_loss.item()
            test_metrics['reg_loss'] += cur_reg_loss.item()
            test_metrics['rank_loss'] += cur_rank_loss.item()
            
            cur_test_pred[:, i] = cur_rr[:, 0].cpu()
            cur_test_gt[:, i] = ground_truth[0]
            cur_test_mask[:, i] = mask[0]
            i += 1
            
        for key in test_metrics:
            test_metrics[key] /= (len(test_loader) + 1)

        cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
    
    return test_metrics, cur_test_perf
