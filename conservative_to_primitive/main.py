

import torch

from data_loader import DataLoader
from trainer import ModelTrainer
from fnn import FNN_Small, DAIN_GRMHD
from cuda_check import CUDAHandler
import numpy as np
from hyperparam_optimization import HyperparameterOptimizer

def run_dain():
    print("Running dain...")
    model = DAIN_GRMHD(input_dim=input_size, output_dim=output_size, num_blocks=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    trainer = ModelTrainer(model, device, scheduler=scheduler, optimizer=optimizer)
    trained_model, train_loss_list, val_loss_list = trainer.train(X_train_scaled, y_train_scaled, X_val_scaled,
                                                                  y_val_scaled, no_of_epochs=5, batch_size=32)

    trainer.test_model(*trainer.duplicate_data(X, y))

def run_basic():
    print("Running basic...")
    model = FNN_Small(input_size, output_size).to(device)

    trainer = ModelTrainer(model, device)
    trained_model, train_loss_list, val_loss_list = trainer.train(X_train_scaled, y_train_scaled, X_val_scaled,
                                                                  y_val_scaled, no_of_epochs=5, batch_size=64)

    # trainer.save_model("fnn_small.pth")

    y_pred, y_actual = trainer.test_model(X_test_scaled, y_test_scaled)
    # trainer.test_model(*trainer.duplicate_data(X, y))

def run_hyperparameter_tuning():
    print("Running hyperparameter tuning...")
    optimizer = HyperparameterOptimizer(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, input_size, output_size)
    study = optimizer.optimize(n_trials=1)
    print(study.best_trial)


cuda_handler = CUDAHandler(use_cuda=True, cuda_device="cuda:0")
device = cuda_handler.check_cuda()

folder_path_conserved = '../Balsara1_shocktube/Balsara1_shocktube_xdir_WZ/output-0000/Balsara1_shocktube_xdir/conservatives/'
conserved = ['db', 'dens', 'mom', 'tau']
prefix_conserved = 'asterx'

folder_path_primitive = '../Balsara1_shocktube/Balsara1_shocktube_xdir_WZ/output-0000/Balsara1_shocktube_xdir/primitives/'
primitive = ["rho", "vel", "eps", "press", "bvec"]
prefix_primitive = 'hydrobasex'


data_loader = DataLoader(folder_path_conserved, conserved, prefix_conserved, folder_path_primitive, primitive, prefix_primitive)
data_loader.load_conserved()
data_loader.load_primitive()

db, dens, mom, tau = data_loader.data_dict['db'], data_loader.data_dict['dens'], data_loader.data_dict['mom'], data_loader.data_dict['tau']
rho, vel, eps, press, bvec = data_loader.data_dict['rho'], data_loader.data_dict['vel'], data_loader.data_dict['eps'], data_loader.data_dict['press'], data_loader.data_dict['bvec']


merged_data = data_loader.merge_data((db, dens, mom, tau), (rho, vel, eps, press, bvec))
merged_data.drop(columns=['iteration', 'time', 'patch', 'level', 'i', 'j', 'k', 'x', 'y', 'z'], inplace=True)

conserved_vars = ['dBx', 'dBy', 'dBz', 'dens', 'momx', 'momy', 'momz', 'tau']  # all conserved
primitive_vars = ['rho', 'eps', 'press', 'velx', 'vely', 'velz', 'Bvecx', 'Bvecy', 'Bvecz']

X = merged_data[conserved_vars]
y = merged_data[primitive_vars]

data_loader.preprocess_data(X, y)

X_train_scaled = data_loader.X_train_scaled
y_train_scaled = data_loader.y_train_scaled
X_val_scaled = data_loader.X_val_scaled
y_val_scaled = data_loader.y_val_scaled
X_test_scaled = data_loader.X_test_scaled
y_test_scaled = data_loader.y_test_scaled

input_size = X_train_scaled.shape[1]
output_size = y_train_scaled.shape[1]

run_basic()
# run_dain()
# run_hyperparameter_tuning()