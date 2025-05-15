import optuna
import concurrent.futures
from trainer import ModelTrainer
from fnn import FNN_Dynamic
import torch
from cuda_check import CUDAHandler

class HyperparameterOptimizer:
    def __init__(self,X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, input_size, output_size):
        self.study = None
        self.cuda_handler = CUDAHandler(use_cuda=True, cuda_device="cuda:0")
        self.device = self.cuda_handler.check_cuda()
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_val_scaled = y_val_scaled
        self.input_size = input_size
        self.output_size = output_size


    def hyper_param_train(self, model, X_train, y_train, X_val, y_val, no_of_epochs=85, batch_size=32, lr=3e-4,
                          patience=5, factor=0.5, stream=None):
        model.to(self.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        with torch.cuda.stream(stream):
            for epoch in range(no_of_epochs):
                model.train()
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i + batch_size]
                    batch_y = y_train_tensor[i:i + batch_size]

                    optimizer.zero_grad()
                    y_pred = model(batch_X)
                    loss = criterion(y_pred, batch_y)

                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_val_tensor)
                    val_loss = criterion(y_val_pred, y_val_tensor)
                scheduler.step(val_loss)
                if epoch % 10 == 0 or epoch == no_of_epochs - 1:
                    print(
                        f'Epoch {epoch}/{no_of_epochs}, Loss: {loss.item():.4e}, Validation Loss: {val_loss.item():.4e}')

        return val_loss.item()

    def objective(self, trial):
        first_layer = trial.suggest_int("n_units_l0", 100, 700, step=50)
        second_layer = trial.suggest_int("n_units_l1", 50, first_layer - 50, step=50)
        hidden_layers = [first_layer, second_layer]

        lr = trial.suggest_categorical("lr", [3e-4, 1e-4, 5e-4])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        no_of_epochs = trial.suggest_int("no_of_epochs", 1, 2)
        patience = trial.suggest_int("patience", 4, 8)
        factor = trial.suggest_categorical("factor", [0.3, 0.5, 0.7])
        model = FNN_Dynamic(self.input_size, self.output_size, hidden_layers)

        stream = torch.cuda.Stream()

        def train_thread():
            return self.hyper_param_train(model, self.X_train_scaled, self.y_train_scaled, self.X_val_scaled, self.y_val_scaled,
                               no_of_epochs=no_of_epochs,
                               batch_size=batch_size,
                               lr=lr,
                               patience=patience,
                               factor=factor,
                               stream=stream)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(train_thread)
            return future.result()

    def optimize(self, n_trials=10):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=n_trials)
        return self.study

