# Located in train/trainer.py
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve

class SiameseTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, scheduler, device, 
                 checkpoint_dir, model_dir, log_file):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.log_file = log_file

        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.positive_distances = []
        self.negative_distances = []
        self.lr_history = []
        self.best_test_eer = float('inf')

    def log_info(self, message):
        if torch.distributed.get_rank() == 0:
            print(message)
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')

    def calculate_accuracy(self, distances, labels, threshold=1):
        return np.mean((distances <= threshold) == (labels == 1))

    def calculate_eer(self, distances, labels):
        try:
            if len(distances) == 0 or len(labels) == 0:
                print("[Warning] distances or labels are empty.")
                return 1.0, 0.0
            unique_labels = np.unique(labels)
            if set(unique_labels) not in [{0, 1}, {-1, 1}]:
                labels = np.where(labels > 0, 1, 0)
            if len(np.unique(labels)) < 2:
                print("[Warning] Labels must contain both positive and negative classes.")
                return 1.0, 0.0
            fpr, tpr, thresholds = roc_curve(labels, -distances, pos_label=1)
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.abs(fnr - fpr))
            eer = (fnr[eer_index] + fpr[eer_index]) / 2
            eer_threshold = thresholds[eer_index]
            return eer, eer_threshold
        except Exception as e:
            print(f"[Warning] Error occurred during EER calculation: {str(e)}")
            return 1.0, 0.0

    def train_epoch(self, epoch, epochs, threshold):
        start_time = time.time()
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pos_dist_sum = 0.0
        neg_dist_sum = 0.0
        pos_count = 0
        neg_count = 0
        train_distances = []
        train_labels_all = []

        scaler = torch.cuda.amp.GradScaler()
        with tqdm(total=len(self.train_loader), desc=f"Training Epoch {epoch+1}/{epochs}", unit='batch', disable=torch.distributed.get_rank() != 0) as pbar:
            for batch_idx, (x1_batch, x2_batch, labels_batch, *_ ) in enumerate(self.train_loader):
                x1_batch, x2_batch, labels_batch = x1_batch.to(self.device), x2_batch.to(self.device), labels_batch.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    output1, output2 = self.model(x1_batch, x2_batch)
                    loss = self.criterion(output1, output2, labels_batch)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item()

                distances = F.pairwise_distance(output1, output2).detach().cpu().numpy()
                train_distances.extend(distances)
                train_labels_all.extend(labels_batch.detach().cpu().numpy())

                accuracy = self.calculate_accuracy(distances, labels_batch.cpu().numpy(), threshold=threshold)
                correct += accuracy * len(labels_batch)
                total += len(labels_batch)

                pos_dist_sum += np.sum(distances[labels_batch.cpu().numpy() == 1])
                neg_dist_sum += np.sum(distances[labels_batch.cpu().numpy() == 0])
                pos_count += np.sum(labels_batch.cpu().numpy() == 1)
                neg_count += np.sum(labels_batch.cpu().numpy() == 0)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)

                if torch.distributed.get_rank() == 0:
                    pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Accuracy': correct / total})
                    pbar.update(1)

        avg_train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_train_loss)
        avg_train_accuracy = correct / total
        self.train_accuracies.append(avg_train_accuracy)
        epoch_time = time.time() - start_time

        avg_pos_dist = pos_dist_sum / pos_count if pos_count > 0 else 0
        avg_neg_dist = neg_dist_sum / neg_count if neg_count > 0 else 0
        self.positive_distances.append(avg_pos_dist)
        self.negative_distances.append(avg_neg_dist)

        train_distances_np = np.array(train_distances)
        train_labels_all_np = np.array(train_labels_all)
        if np.isnan(train_distances_np).any():
            valid_mask = ~np.isnan(train_distances_np)
            train_distances_np = train_distances_np[valid_mask]
            train_labels_all_np = train_labels_all_np[valid_mask]

        train_eer, train_eer_threshold = self.calculate_eer(train_distances_np, train_labels_all_np)

        return {
            'epoch_time': epoch_time,
            'avg_train_loss': avg_train_loss,
            'avg_train_accuracy': avg_train_accuracy,
            'avg_pos_dist': avg_pos_dist,
            'avg_neg_dist': avg_neg_dist,
            'train_eer': train_eer,
            'train_eer_threshold': train_eer_threshold
        }

    def test_epoch(self, epoch, epochs, threshold):
        test_loss = 0.0
        correct = 0
        total = 0
        test_distances = []
        test_labels_all = []
        target_fars = [0.001, 0.0001, 0.000029]

        self.model.eval()
        with tqdm(total=len(self.test_loader), desc=f"Testing Epoch {epoch+1}/{epochs}", unit='batch', disable=torch.distributed.get_rank() != 0) as pbar_test:
            with torch.no_grad():
                for test_batch_idx, (test_x1, test_x2, test_labels, *_) in enumerate(self.test_loader):
                    test_x1, test_x2, test_labels = test_x1.to(self.device), test_x2.to(self.device), test_labels.to(self.device)
                    output1, output2 = self.model(test_x1, test_x2)
                    test_batch_loss = self.criterion(output1, output2, test_labels)
                    test_loss += test_batch_loss.item()
                    distances = F.pairwise_distance(output1, output2).cpu().numpy()
                    test_distances.extend(distances)
                    test_labels_all.extend(test_labels.cpu().numpy())

                    accuracy = self.calculate_accuracy(distances, test_labels.cpu().numpy(), threshold=threshold)
                    correct += accuracy * len(test_labels)
                    total += len(test_labels)

                    if torch.distributed.get_rank() == 0:
                        pbar_test.set_postfix({'Test Loss': test_loss / (test_batch_idx + 1), 'Test Accuracy': correct / total})
                        pbar_test.update(1)

        avg_test_loss = test_loss / len(self.test_loader)
        self.test_losses.append(avg_test_loss)
        avg_test_accuracy = correct / total
        self.test_accuracies.append(avg_test_accuracy)

        test_distances_np = np.array(test_distances)
        test_labels_np = np.array(test_labels_all)
        if np.isnan(test_distances_np).any():
            valid_mask = ~np.isnan(test_distances_np)
            test_distances_np = test_distances_np[valid_mask]
            test_labels_np = test_labels_np[valid_mask]

        test_eer, test_eer_threshold = self.calculate_eer(test_distances_np, test_labels_np)

        frrs = {}
        try:
            fpr, tpr, thresholds = roc_curve(test_labels_np, -test_distances_np, pos_label=1)
            fnr = 1 - tpr
            for target_far in target_fars:
                idx = np.where(fpr <= target_far)[0]
                if len(idx) > 0:
                    best_idx = idx[-1]
                    threshold_val = thresholds[best_idx]
                    frr = fnr[best_idx]
                    frrs[f"FAR {target_far * 100:.3f}%"] = {"Threshold": threshold_val, "FRR (%)": frr * 100}
                else:
                    frrs[f"FAR {target_far * 100:.3f}%"] = {"Threshold": None, "FRR (%)": None}
        except Exception:
            frrs = {f"FAR {far * 100:.3f}%": {"Threshold": None, "FRR (%)": None} for far in target_fars}

        return {
            'avg_test_loss': avg_test_loss,
            'avg_test_accuracy': avg_test_accuracy,
            'test_eer': test_eer,
            'test_eer_threshold': test_eer_threshold,
            'frrs': frrs
        }

    def save_model_checkpoint(self, epoch, threshold):
        if torch.distributed.get_rank() == 0:
            path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'test_eer_threshold': threshold
            }, path)

    def save_best_model(self, epoch, eer, threshold):
        if eer < self.best_test_eer and torch.distributed.get_rank() == 0:
            self.best_test_eer = eer
            path = os.path.join(self.model_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'test_eer_threshold': threshold
            }, path)
            self.log_info(f"Best model saved with EER: {eer:.4f} at {path}")

    def resume_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0) + 1, checkpoint.get('test_eer_threshold', 1.0)

    def run_training(self, epochs):
        threshold = 1.0
        for epoch in range(epochs):
            train_result = self.train_epoch(epoch, epochs, threshold)
            test_result = self.test_epoch(epoch, epochs, threshold)
            threshold = test_result['test_eer_threshold']

            if torch.distributed.get_rank() == 0:
                self.log_info(f"\nEpoch [{epoch+1}/{epochs}]")
                self.log_info(f"Train Loss: {train_result['avg_train_loss']:.4f}, Acc: {train_result['avg_train_accuracy']:.6f}")
                self.log_info(f"Test  Loss: {test_result['avg_test_loss']:.4f}, Acc: {test_result['avg_test_accuracy']:.6f}")
                self.log_info(f"EER: {test_result['test_eer']:.4f}, Threshold: {threshold:.4f}")
                for key, val in test_result['frrs'].items():
                    self.log_info(f"{key} - Threshold: {val['Threshold']}, FRR: {val['FRR (%)']}%")
                self.save_model_checkpoint(epoch, threshold)

            self.scheduler.step(test_result['avg_test_loss'])

        if torch.distributed.get_rank() == 0:
            self.save_best_model(epoch, test_result['test_eer'], threshold)
            self.log_info("===== Training finished =====")
            self.log_info(f"Best Testing EER: {self.best_test_eer:.4f}")
