import torch
import time
import numpy as np

class ModelTrainer:
    def __init__(self, model, device, criterion=None, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion or torch.nn.MSELoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=3e-4)
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=6, factor=0.3)

    def train(self, X_train, y_train, X_val, y_val, no_of_epochs=85, batch_size=32):
        train_losses = []
        val_losses = []
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        print(f"Model is on: {next(self.model.parameters()).device}")
        print(f"X_train_tensor is on: {X_train_tensor.device}")

        for epoch in range(no_of_epochs):
            self.model.train()
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]

                self.optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = self.criterion(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model(X_val_tensor)
                val_loss = self.criterion(y_val_pred, y_val_tensor)
            self.scheduler.step(val_loss)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

            if epoch % 10 == 0 or epoch == no_of_epochs - 1:
                print(f'Epoch {epoch}/{no_of_epochs}, Loss: {loss.item():.4e}, Validation Loss: {val_loss.item():.4e}')

        return self.model, train_losses, val_losses

    def save_model(self, save_path):
        scripted_model = torch.jit.script(self.model)

        scripted_model.save(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        scripted_model = torch.jit.load(load_path)

        self.model = scripted_model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {load_path}")

    def test_model(self, X_test, y_test, n_runs=1):
        device = self.device
        print(
            f"Testing on device: {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "Testing on CPU")

        self.model.to(device)
        self.model.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        total_time = 0
        for run in range(n_runs):
            torch.cuda.synchronize()  # Ensure all previous GPU operations are complete
            start_time = time.perf_counter()

            with torch.no_grad():
                y_pred = self.model(X_test_tensor)
                test_loss = self.criterion(y_pred, y_test_tensor)

            torch.cuda.synchronize()  # Ensure GPU operations are finished
            end_time = time.perf_counter()

            run_time = end_time - start_time
            total_time += run_time

            if run == n_runs - 1:
                print(f"Run {run + 1}/{n_runs}: Test Loss: {test_loss.item():.4e}")
                print(f"Run Time: {run_time:.6f} seconds")

        avg_time_per_run = total_time / n_runs
        avg_time_per_sample = avg_time_per_run / len(X_test_tensor)

        print(f"Total Samples: {len(X_test_tensor)}")
        print(f"Average Inference Time: {avg_time_per_run:.6f} seconds")
        print(f"Average Inference Time per Sample: {avg_time_per_sample * 1000:.6f} ms")

        return y_pred.cpu().numpy(), y_test_tensor.cpu().numpy()

    def duplicate_data(self,X, y, size= 1000000):
        req_size = size
        num_samples = len(X)

        # Calculate how many times to repeat the data to get at least 1,000,000 samples
        repeat_factor = int(np.ceil(req_size / num_samples))

        # Repeat X and y along the sample axis using np.tile
        X_large = np.tile(X, (repeat_factor, 1))  # Repeat X values
        y_large = np.tile(y, (repeat_factor, 1))  # Repeat y values

        # Ensure the final size is exactly 1,000,000
        X_large = X_large[:req_size]
        y_large = y_large[:req_size]

        return X_large, y_large





