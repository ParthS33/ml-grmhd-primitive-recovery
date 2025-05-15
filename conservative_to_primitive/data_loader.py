import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, folder_path_conserved, conserved, prefix_conserved, folder_path_primitive, primitive, prefix_primitive, sep="\t"):
        self.folder_path_conserved = folder_path_conserved
        self.folder_path_primitive = folder_path_primitive
        self.conserved_var = conserved
        self.primitive_var = primitive
        self.prefix_primitive = prefix_primitive
        self.prefix_conserved = prefix_conserved
        self.sep = sep
        self.data_dict = {}

        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_val_scaled = None
        self.y_test_scaled = None
        self.scaler_X = None
        self.scaler_y = None

    def load_data(self, variables, prefix, folder_path):
        for var in variables:
            file_pattern = f'{prefix}-{var}*.tsv'
            file_list = glob.glob(os.path.join(folder_path, file_pattern))

            df_list = [pd.read_csv(file, sep=self.sep) for file in file_list]
            if df_list:
                self.data_dict[var] = pd.concat(df_list, ignore_index=True)
                self.data_dict[var].columns = [col.split(':')[-1].split('-')[-1] for col in self.data_dict[var].columns]
                self.data_dict[var] = self.data_dict[var].drop_duplicates()
            else:
                print(f"Warning: No files found for {var}")

        for var, df in self.data_dict.items():
            print(f"{var} - Missing Values:\n{df.isnull().sum()}\n")
        print()
        # return tuple(self.data_dict.values())

    def load_conserved(self):
        self.load_data(self.conserved_var, self.prefix_conserved, self.folder_path_conserved)

    def load_primitive(self):
        self.load_data(self.primitive_var, self.prefix_primitive, self.folder_path_primitive)

    def merge_data(self, conserved, primitive,
                   keys=['iteration', 'x', 'y', 'z', 'time', 'patch', 'level', 'i', 'j', 'k']):
        db, dens, mom, tau = conserved
        rho, vel, eps, press, bvec = primitive

        all_data = [db, dens, mom, tau, rho, vel, eps, press, bvec]

        merged_df = all_data[0]
        for df in all_data[1:]:
            merged_df = pd.merge(merged_df, df, on=keys, how='inner')

        return merged_df

    def split_data(self, X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=random_state)

        val_relative_size = val_size / (val_size + test_size)

        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_relative_size),
                                                        random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.X_val_scaled = self.scaler_X.transform(X_val)
        self.X_test_scaled = self.scaler_X.transform(X_test)
        self.y_train_scaled = self.scaler_y.fit_transform(y_train)
        self.y_val_scaled = self.scaler_y.transform(y_val)
        self.y_test_scaled = self.scaler_y.transform(y_test)

    def preprocess_data(self, X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, train_size, val_size, test_size,
                                                                         random_state)
        self.scale_data(X_train, X_val, X_test, y_train, y_val, y_test)
