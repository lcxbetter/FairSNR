import argparse
import torch
import numpy as np
import torch.nn.functional as F
from data_utils import get_dataset
from eval import evaluate_ged3
from utils import seed_everything, get_enc_cls_opt
from models import Discriminator,grad_reverse,VGAE
from var import SNLoss

def run(data, args):
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    data = data.to(args.device)
    loss_fn = SNLoss(num_classes=2, int_lambda=0.01, int_reg=0.02, target_lambda= 0.2, kl_lambda=0.5, bias=1, prior_type='conditional')
    encoder, classifier,Intervenor, optimizer_e, optimizer_c,optimizer_i = get_enc_cls_opt(args)
    train_mask = data.train_mask
    y = data.y[train_mask].unsqueeze(1).to(args.device).long()

    encoder_mv = VGAE(args).to(args.device)
    optimizer_mv = torch.optim.Adam(encoder_mv.parameters(), lr=0.01)
    discriminator = Discriminator(input_dim=args.hidden).to(args.device)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)  
    for count in range(args.runs):
        seed_everything(count + args.seed)
        classifier.reset_parameters()
        encoder.reset_parameters()
        Intervenor.reset_parameters()
        encoder_mv.reset_parameters()
        discriminator.reset_parameters()
        best_val_tradeoff = 0
        for epoch in range(args.epochs):
            optimizer_d.zero_grad()
            encoder.eval() 
            with torch.no_grad():
                z_detach = encoder(data.x, data.edge_index)
            pred_s = discriminator(z_detach)
            loss_disc = F.binary_cross_entropy_with_logits(
                pred_s[train_mask], data.sens[train_mask].unsqueeze(1).float())
            loss_disc.backward()
            optimizer_d.step()

    
            encoder.train()
            classifier.train()
            encoder_mv.train()
            Intervenor.train()

            optimizer_e.zero_grad()
            optimizer_c.zero_grad()
            optimizer_mv.zero_grad()

            z = encoder(data.x, data.edge_index)
            m, v = encoder_mv(data.x, data.edge_index)


            adv_pred = discriminator(grad_reverse(z, alpha=args.alpha_d))
            loss_adv = F.binary_cross_entropy_with_logits(adv_pred[train_mask], data.sens[train_mask].unsqueeze(1).float())
            z_c = Intervenor(z)
            int_z = z + z_c

            out_real = classifier(z)
            out_int = classifier(int_z)

            loss_main = loss_fn(z, int_z, m, v,
                                y, out_real[train_mask],
                                out_int[train_mask], z_c,
                                train_mask, turn='min')

            loss = (loss_main + args.lambda_adv * loss_adv)

            loss.backward()
            optimizer_e.step()
            optimizer_c.step()
            optimizer_mv.step()

            if epoch % args.warmup == 0:
                optimizer_i.zero_grad()
                encoder.eval()
                classifier.eval()
                encoder_mv.eval()

                with torch.no_grad():
                    z_frozen, (m_frozen, v_frozen) = encoder(data.x, data.edge_index), \
                        encoder_mv(data.x, data.edge_index)

                intervention = Intervenor(z_frozen)
                int_z_frozen = z_frozen + intervention
                out_real = classifier(z_frozen)
                out_int = classifier(int_z_frozen)

                loss_interv = loss_fn(
                    z_frozen, int_z_frozen, m_frozen, v_frozen,
                    y, out_real[train_mask], out_int[train_mask],
                    intervention, train_mask, turn='max')
                loss_interv.backward()
                optimizer_i.step()

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(classifier, encoder, data)

            if epoch % 10 == 0:
                print(
                    "RUN: {}/{}, Epoch: {:04}/{:04} | Val Acc: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test F1: {:.4f}, Test SP: {:.4f}, Test EO: {:.4f}".format(
                        count + 1, args.runs, epoch, args.epochs, accs['val'], accs['test'], auc_rocs['test'],
                        F1s['test'], tmp_parity['test'], tmp_equality['test']
                    ))

            if (auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                    tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff):
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                                    accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])

                print(
                    "\033[0;30;41m RUN: {}/{}, Epoch: {:04}/{:04} | Val Acc: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test F1: {:.4f}, Test SP: {:.4f}, Test EO: {:.4f}\033[0m".format(
                        count + 1, args.runs, epoch, args.epochs, accs['val'], accs['test'], auc_rocs['test'],
                        F1s['test'], tmp_parity['test'], tmp_equality['test']
                    ))

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

    return acc, f1, auc_roc, parity, equality


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--i_lr', type=float, default=0.01)
    parser.add_argument('--i_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--d_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=18)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--alpha_d', type=float, default=0.01)
    parser.add_argument('--lambda_adv', type=float, default=0.5)
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data.x.shape[1], 1

    acc, f1, auc_roc, parity, equality = run(data,args)
    print('======' + args.dataset + args.encoder + '======')
    print('auc_roc: {:.2f} +- {:.2f}'.format(np.mean(auc_roc) * 100, np.std(auc_roc) * 100))
    print('Acc: {:.2f} +- {:.2f}'.format(np.mean(acc) * 100, np.std(acc) * 100))
    print('f1: {:.2f} +- {:.2f}'.format(np.mean(f1) * 100, np.std(f1) * 100))
    print('parity: {:.2f} +- {:.2f}'.format(np.mean(parity) * 100, np.std(parity) * 100))
    print('equality: {:.2f} +- {:.2f}'.format(np.mean(equality) * 100, np.std(equality) * 100))
