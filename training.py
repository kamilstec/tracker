from sacred import Experiment
from sacred.observers import FileStorageObserver
import os.path as osp
import os
import numpy as np
import time
import torch
from torch_geometric.data import DataLoader
import torch.utils.tensorboard as tb

from lib.datasets.config import get_output_dir, get_tb_dir
from lib.datasets.datasets import Datasets
from lib.network.complete_net import completeNet
from lib.network.utils import weighted_binary_cross_entropy, hungarian

ex = Experiment('Training network')

ex.observers.append(FileStorageObserver('../gdrive/MyDrive/KamMOT_trenowanie/logs'))

ex.add_config('config/training.yaml')

@ex.automain
def main(_config, _log, _run, training):

    torch.manual_seed(training['seed'])
    torch.cuda.manual_seed(training['seed'])
    np.random.seed(training['seed'])
    torch.backends.cudnn.deterministic = True

    print(_config)

    output_dir = osp.join(get_output_dir(training['cnn_encoder']) + '_runID_' + _run._id)
    tb_dir = osp.join(get_tb_dir(training['cnn_encoder']) + '_runID_' + _run._id)
    tb_dir_val = tb_dir + '_val'
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    if not osp.exists(tb_dir):
        os.makedirs(tb_dir)
    if not osp.exists(tb_dir_val):
        os.makedirs(tb_dir_val)

    writer = tb.SummaryWriter(tb_dir)
    writer_val = tb.SummaryWriter(tb_dir_val)

    _log.info(f"Building network with CNN encoder: {training['cnn_encoder']}")

    model = completeNet(cnn_encoder=training['cnn_encoder'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _log.info("Initializing datasets")

    train_dataset = Datasets(training['train_dataset'], training['train_dataloader'])
    val_dataset = Datasets(training['val_dataset'], training['val_dataloader'])
    train_dataset_len = 0
    val_dataset_len = 0
    for seq in train_dataset:
        train_dataset_len += len(seq)
    for seq in val_dataset:
        val_dataset_len += len(seq)
    _log.info('Train dataset size: ' + str(train_dataset_len) + ' | Val dataset size: ' + str(val_dataset_len))

    num_epochs = training['solver']['epochs']
    lr = training['solver']['lr']
    wd = training['solver']['weight_decay']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    weights = [10, 1]

    best_val_loss = float('inf')
    best_epoch = 0
    _log.info("Training...")
    for epoch in range(1, num_epochs + 1):
        tick_epoch = time.time()  # tik tok - liczenie czasu

        epoch_total_train = 0
        epoch_total_ones_train = 0
        epoch_total_zeros_train = 0
        epoch_correct_train = 0
        epoch_correct_ones_train = 0
        epoch_correct_zeros_train = 0
        running_loss_train = 0

        epoch_total_val = 0
        epoch_total_ones_val = 0
        epoch_total_zeros_val = 0
        epoch_correct_val = 0
        epoch_correct_ones_val = 0
        epoch_correct_zeros_val = 0
        running_loss_val = 0

        iterations_train = 0
        iterations_val = 0

        for seq_idx, seq in enumerate(train_dataset, start=1):
            seq_val = val_dataset[seq_idx-1]

            tick_seq = time.time()
            Sequence_Training_Loss = 0
            Sequence_Val_Loss = 0

            train_loader = DataLoader(seq, batch_size=1, num_workers=2, shuffle=True)
            val_loader = DataLoader(seq_val, batch_size=1, num_workers=2, shuffle=True)
            #print(len(val_loader))

            model.train()
            for batch_idx, batch in enumerate(train_loader, start=1):
                tick_batch = time.time()
                iterations_train += 1

                batch = batch.to(device)

                optimizer.zero_grad()

                batch_total = 0
                batch_total_ones = 0  # ilość doapsowań ścieżka-detekcja w tym batchu
                batch_total_zeros = 0  # ilość niedopasowań pomiędzy każdą parą ścieżka i detekcja
                batch_correct = 0
                batch_correct_ones = 0  # ilość prawidłowych dopasowań ścieżka-detekcja
                batch_correct_zeros = 0  # ilość prawidłowych niedopasowań ścieżka-detekcja

                output, output2, ground_truth, ground_truth2, det_num, tracklet_num = model(batch)
                loss = weighted_binary_cross_entropy(output, ground_truth, weights)
                loss.backward()
                optimizer.step()
                cleaned_output = hungarian(output2, det_num, tracklet_num)
                running_loss_train += loss.item() # loss dla epoki
                Sequence_Training_Loss += loss.item()

                batch_total += cleaned_output.size(0)
                ones = torch.tensor([1 for x in cleaned_output]).to(device)
                zeros = torch.tensor([0 for x in cleaned_output]).to(device)
                batch_total_ones += (cleaned_output == ones).sum().item()
                batch_total_zeros += (cleaned_output == zeros).sum().item()
                batch_correct += (cleaned_output == ground_truth2).sum().item()
                temp1 = (cleaned_output == ground_truth2)
                temp2 = (cleaned_output == ones)
                batch_correct_ones += (temp1 & temp2).sum().item()
                temp3 = (cleaned_output == zeros)
                batch_correct_zeros += (temp1 & temp3).sum().item()

                epoch_total_train += batch_total
                epoch_total_ones_train += batch_total_ones
                epoch_total_zeros_train += batch_total_zeros
                epoch_correct_train += batch_correct
                epoch_correct_ones_train += batch_correct_ones
                epoch_correct_zeros_train += batch_correct_zeros
                tock_batch = time.time()
                #print('Epoch [%d/%d] | Seq: %s | Batch: [%d/%d] | Training_Loss: %.3f | %.3f s/batch |' %
                #      (epoch,num_epochs, seq._seq_name, batch_idx,len(train_loader), loss.item(),
                #      tock_batch-tick_batch))
            tock_seq = time.time()
            print('Epoch [%d/%d] | Seq: %s | Sequence_Training_Loss: %.3f | %.3f s/seq |' %
                  (epoch, num_epochs, seq.seq_name, Sequence_Training_Loss / len(train_loader), tock_seq - tick_seq))
            validation_epoch=1
            if epoch%validation_epoch==0:
                model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader, start=1):
                        batch = batch.to(device)
                        iterations_val += 1

                        batch_total = 0
                        batch_total_ones = 0
                        batch_total_zeros = 0
                        batch_correct = 0
                        batch_correct_ones = 0
                        batch_correct_zeros = 0

                        output, output2, ground_truth, ground_truth2, det_num, tracklet_num = model(batch)

                        loss = weighted_binary_cross_entropy(output, ground_truth, weights)

                        running_loss_val += loss.item()
                        Sequence_Val_Loss += loss.item()

                        cleaned_output = hungarian(output2, det_num, tracklet_num)
                        batch_total += cleaned_output.size(0)
                        ones = torch.tensor([1 for x in cleaned_output]).to(device)
                        zeros = torch.tensor([0 for x in cleaned_output]).to(device)
                        batch_total_ones += (cleaned_output == ones).sum().item()
                        batch_total_zeros += (cleaned_output == zeros).sum().item()
                        batch_correct += (cleaned_output == ground_truth2).sum().item()
                        temp1 = (cleaned_output == ground_truth2)
                        temp2 = (cleaned_output == ones)
                        batch_correct_ones += (temp1 & temp2).sum().item()
                        temp3 = (cleaned_output == zeros)
                        batch_correct_zeros += (temp1 & temp3).sum().item()
                        epoch_total_val += batch_total
                        epoch_total_ones_val += batch_total_ones
                        epoch_total_zeros_val += batch_total_zeros
                        epoch_correct_val += batch_correct
                        epoch_correct_ones_val += batch_correct_ones
                        epoch_correct_zeros_val += batch_correct_zeros
                    tock_seq = time.time()
                    print('Epoch [%d/%d] | Seq: %s | Sequence_Val_Loss: %.3f | %.3f s/seq |' %
                          (epoch, num_epochs, seq.seq_name, Sequence_Val_Loss / len(val_loader), tock_seq - tick_seq))
        # podsumowanie całej epoki:
        tock_epoch = time.time()
        print('Epoch [%d/%d] | Epoch_Train_Loss: %.3f | Epoch_Val_Loss: %.3f | %.3f s/epoch || '
              'Epoch_Total_Train_Accuracy: %.3f | Ones_Train_Accuracy: %.3f | Zeros_Train_Accuracy: %.3f || '
              'Epoch_Total_Val_Accuracy: %.3f | Ones_Val_Accuracy: %.3f | Zeros_Val_Accuracy: %.3f |'
              % (epoch, num_epochs, running_loss_train / iterations_train, running_loss_val / iterations_val,
                 tock_epoch - tick_epoch,
                 100 * epoch_correct_train / epoch_total_train, 100 * epoch_correct_ones_train / epoch_total_ones_train,
                 100 * epoch_correct_zeros_train / epoch_total_zeros_train,
                 100 * epoch_correct_val / epoch_total_val, 100 * epoch_correct_ones_val / epoch_total_ones_val,
                 100 * epoch_correct_zeros_val / epoch_total_zeros_val))

        filename = training['cnn_encoder'] + '_gcnn_' + str(epoch) + '.pth'
        filename = os.path.join(output_dir, filename)
        torch.save({
            'cnn_encoder': training['cnn_encoder'],  # użyty encoderCNN
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
            'loss': running_loss_val / iterations_val,  # średnia strata na każdym batchu walidacyjnym
        }, filename)
        if (running_loss_val/iterations_val) < best_val_loss:
            print('[*] Best model')
            best_val_loss = (running_loss_val/iterations_val)
            best_epoch = epoch
            # tylko jeden najlepszy model
            filename = training['cnn_encoder'] + '_gcnn_best' + '.pth'
            filename = os.path.join(output_dir, filename)
            torch.save({
                        'cnn_encoder': training['cnn_encoder'],  # użyty encoderCNN
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss_val/iterations_val,  # średnia strata na każdym batchu walidacyjnym
                        }, filename)
            print('Wrote snapshot to: {:s}'.format(filename))

        writer.add_scalar('Loss', running_loss_train / iterations_train, epoch)
        writer_val.add_scalar('Loss', running_loss_val / iterations_val, epoch)
        writer.add_scalar('Total accuracy', 100*epoch_correct_train/epoch_total_train, epoch)
        writer_val.add_scalar('Total accuracy', 100*epoch_correct_val/epoch_total_val, epoch)
        writer.add_scalar('BBA', 100*0.5*(epoch_correct_ones_train/epoch_total_ones_train +
                                                epoch_correct_zeros_train/epoch_total_zeros_train), epoch)
        writer_val.add_scalar('BBA', 100*0.5*(epoch_correct_ones_val/epoch_total_ones_val +
                                                epoch_correct_zeros_val/epoch_total_zeros_val), epoch)

        print('Koniec epoki ' + '#'*60)
    writer.close()
    print('Best epoch: ' + str(best_epoch))
    print('Best loss: ' + str(best_val_loss))
    print('[*] FINISH TRAIN')
