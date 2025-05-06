import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import MinMaxScaler
import plotting_routine as plotting
import json
from co_util import assign_CO_flag, assign_bin_no, assign_one_hot_encoded_bin_no, rebalance_bins, generate_bin, bin_no, convert_bin_to_z
from itertools import repeat

class PDF_MC:
    def __init__(self, **config):
            self.num_input_features = config["num_input_features"]
            self.num_hidden_neurons = config["num_hidden_neurons"]
            self.num_hidden_layers = config["num_hidden_layers"]
            self.num_epochs = config["num_epochs"]
            self.learning_rate = config["learning_rate"]
            self.size_batch = config["size_batch"]
            self.momentum = config["momentum"]

            self.bins = generate_bin(4)
            del self.bins[-2]
            self.num_bins = len(self.bins) - 1
            self.band_col_index = list(range(self.num_input_features))

            self.train_ratio = config["train_ratio"]
            self.evaluation = config["evaluation"]
            self.evaluation_ratio = config["evaluation_ratio"]

            self.CO_ratio = config['CO_ratio']
            self.model_no = config['model_no']
            self.weights_list = config['weights_list']
            self.rebalance = config['rebalance']
            self.rebalance_list = config['rebalance_list']

            self.input_csv_path = config["input_csv_path"]
            self.suffix = f'{self.input_csv_path.stem}_{self.weights_list}_rebalance_{self.rebalance}_{self.rebalance_list}_{self.model_no}'

            if 'output_csv_path' not in config:
                self.output_csv_path = config.get("output_dir_path") / f"results_PDF_MC_{self.suffix}.csv"
            else:
                self.output_csv_path = config.get("output_csv_path")
            
            if 'output_pdf_path' not in config:
                self.output_pdf_path = config.get("output_dir_path") / f"results_PDF_MC_{self.suffix}.pdf"
            else:
                self.output_pdf_path = config.get("output_pdf_path")

            if 'model_path' not in config:
                self.model_path = config.get("model_dir_path") / f"results_PDF_MC_{self.suffix}.pth"
            else:
                self.model_path = config.get("model_path")

    class NeuralNetwork(nn.Module):
        def __init__(self, PDF_MC_model, num_input_features, num_hidden_neurons, num_hidden_layers):
            super(PDF_MC.NeuralNetwork, self).__init__()
            self.input_layer = nn.Linear(num_input_features, num_hidden_neurons)
            self.hidden_layers = nn.ModuleList([nn.Linear(num_hidden_neurons, num_hidden_neurons) for n in range(num_hidden_layers - 1)])
            self.output_layer = nn.Linear(num_hidden_neurons, PDF_MC_model.num_bins)

        # Define the computation performed in every cell
        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            for hidden_layer in self.hidden_layers:
                x = torch.relu(hidden_layer(x))
            x = self.output_layer(x) # Softmax will be automatically applied in CrossEntropyLoss()
            return x

    def preprocess_data(self):
        raw_df = pd.read_csv(self.input_csv_path)
        # raw_df = raw_df.drop(raw_df.columns[0], axis = 1) # Need to delete after deleting "original index" column in relz_lr.csv

        raw_df = assign_CO_flag(raw_df)

        raw_df = assign_bin_no(raw_df, self.bins)
        raw_df = assign_one_hot_encoded_bin_no(raw_df, self.bins)

        df = raw_df
        if self.evaluation:
            eval_df = raw_df.sample(frac = self.evaluation_ratio)
            df = df.drop(eval_df.index)

        if self.rebalance:
            df = rebalance_bins(df, *self.rebalance_list)

        data = {
            'raw_df': raw_df,
            'df': df, # this will be used to build dataloaders
            'eval_df': eval_df if self.evaluation else None
        }

        return data

    def build_dl(self, data):
        df = data['df']
        eval_df = data['eval_df']

        train_ratio = 1 - self.evaluation_ratio

        # Initialize dataframes
        F = df.iloc[:, self.band_col_index]
        T = df['bin']
        T = np.vstack(T).astype(np.uint8)

        # Initialize tensors from dataframes
        idx_tensor = torch.IntTensor(F.index)
        F_tensor = torch.FloatTensor(F.values)
        T_tensor = torch.FloatTensor(T)

        # Initialize TensorDatasets from tensors
        tensor_data = TensorDataset(idx_tensor, F_tensor, T_tensor)

        train_size = int(len(tensor_data) * self.train_ratio)
        test_size = len(tensor_data) - train_size

        train_data, test_data = random_split(dataset = tensor_data, lengths = [train_size, test_size])

        size_batch = self.size_batch

        # Initialize DataLoaders
        train_dl = DataLoader(dataset = train_data,
                            batch_size = size_batch,
                            shuffle = True)
        test_dl = DataLoader(dataset = test_data,
                            batch_size = size_batch,
                            shuffle = False)
        full_dl = self.build_full_dl(data)["full_dl"]

        if self.evaluation:
            # Initialize Dataframe
            eval_F = df.iloc[:, self.band_col_index]
            eval_T = df['bin']
            eval_T = np.vstack(T).astype(np.uint8)

            # Intialize tensors from Dataframes
            eval_idx_tensor = torch.IntTensor(eval_F.index)
            eval_F_tensor = torch.FloatTensor(eval_F.values)
            eval_T_tensor = torch.FloatTensor(eval_T)

            # Initialize TensorDatasets from tensors
            eval_data = TensorDataset(eval_idx_tensor, eval_F_tensor, eval_T_tensor)

            # Initialize DataLoader
            eval_dl = DataLoader(dataset = eval_data,
                                    batch_size = size_batch,
                                    shuffle = False)

        dl = {
            'train_dl': train_dl,
            'test_dl': test_dl,
            'full_dl': full_dl,
            'eval_dl': eval_dl if self.evaluation else None
        }

        return dl

    def build_full_dl(self, data):
        # need to build from raw_df, not df => need to save raw_df separately
        raw_df = data['raw_df']

        # Initialize dataframes
        raw_z = raw_df['Spec z']
        raw_F = raw_df.iloc[:, self.band_col_index]
        raw_T = raw_df['bin']
        raw_T = np.vstack(raw_T).astype(np.uint8)

        # Initialize tensors from dataframes
        raw_idx_tensor = torch.IntTensor(raw_F.index)
        raw_z_tensor = torch.FloatTensor(raw_z.values)
        raw_F_tensor = torch.FloatTensor(raw_F.values)
        raw_T_tensor = torch.FloatTensor(raw_T)
        full_data = TensorDataset(raw_idx_tensor, raw_F_tensor, raw_T_tensor)

        full_dl = DataLoader(dataset = full_data,
                            batch_size = self.size_batch,
                            shuffle = False)

        dl = {"full_dl": full_dl}

        return dl

    def train_model(self, dl):
        model = PDF_MC.NeuralNetwork(self, self.num_input_features, self.num_hidden_neurons, self.num_hidden_layers)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = self.learning_rate, momentum = self.momentum)

        min_loss = 1
        best_model = None

        num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            for i, inputs, targets in dl["train_dl"]:
                optimizer.zero_grad()
                outputs_pred = model(inputs)
                loss = criterion(outputs_pred, targets)
                loss.backward()
                optimizer.step()

            if loss < min_loss:
                min_loss = loss.item()
                best_model = model.state_dict()
                torch.save(best_model, self.model_path)
                print('best model')

            if ((epoch + 1)%10 == 0) or (loss == min_loss):
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.12f}')
                print()

        return best_model

    def evaluate_model_helper(self, data, dl):
        # Load the best model
        best_model = PDF_MC.NeuralNetwork(self, self.num_input_features, self.num_hidden_neurons, self.num_hidden_layers)
        best_model.load_state_dict(torch.load(self.model_path))
        best_model.eval()

        with torch.no_grad():
            idx = [] # no need for np.ndarray instead of list?
            y_true = []
            y_pred = []

            for i, inputs, targets in dl:
                outputs_pred = best_model(inputs)

                idx.extend(i.numpy())
                y_true.extend(targets.numpy())
                y_pred.extend(outputs_pred.numpy())

        results = pd.DataFrame({
            'idx': pd.Series(idx)
        })

        bin_header = [f'{i}: {self.bins[i]} ~ {self.bins[i+1]}' for i in range(self.num_bins)]
        y_pred_df = pd.DataFrame(y_pred, columns = bin_header)

        softmax = torch.nn.Softmax(dim = 1)
        y_pred_tensor = torch.Tensor(y_pred_df.values)
        y_pred_df = pd.DataFrame(softmax(y_pred_tensor), columns = bin_header, index = idx)

        results = pd.concat([data["raw_df"].iloc[idx, :7], y_pred_df], axis = 1)

        if 'mse' in results.columns: results = results.drop('mse', axis = 1)
        return results

    def evaluate_model(self, data, dl, full_evaluation = False):
        if full_evaluation:
            return self.evaluate_model_helper(data, dl["full_dl"])
        else:
            if self.evaluation:
                return self.evaluate_model_helper(data, dl["eval_dl"])
            else:
                return self.evaluate_model_helper(data, dl['train_dl'])

    def save_results(self, data, results, full_save = True):
        raw_df = data['raw_df']
        idx = results.index.to_numpy()
        bins = self.bins
        num_bins = self.num_bins
        y_pred = results.iloc[:, 7:] # excluding 5-band magnitudes, Spec z, Phot z,
        y_true = np.digitize(raw_df['Spec z'].iloc[idx].to_numpy(), bins) - 1
        y_pred_max = np.array([bin_no(y_pred_row[1]) for y_pred_row in y_pred.iterrows()])

        results['Original bin'] = y_true
        results['Predicted bin'] = y_pred_max
        results['Middle value of the bin'] = np.array([convert_bin_to_z(original_bin, bins) for original_bin in results['Original bin']])
        results['Phot z'] = np.array([convert_bin_to_z(predicted_bin, bins) for predicted_bin in results['Predicted bin']])

        count_dicts = []
        for bin_n in range(num_bins):
            orig_bin_idx = np.where(results['Original bin'] == bin_n)
            unique, counts = np.unique(results.iloc[orig_bin_idx]['Predicted bin'], return_counts = True) # (this is called chain indexing) Neither results[orig_bin_idx] nor results.iloc[orig_bin_idx, 'Predicted bin'] works which is weird
            count_temp = dict(zip([n for n in range(num_bins)], [0 for _ in range(num_bins)]))
            count_temp.update(dict(zip(unique, counts)))
            count_dicts.append(list(count_temp.values()))

        frac_lists = []
        # First column: fractions of correctly identified NOs
        # Second column: fractions of COs
        for bin_n in range(num_bins):
            frac_temp = []
            frac_temp.append(round(count_dicts[bin_n][bin_n] / sum(count_dicts[bin_n])* 100, 2))

            orig_bin_idx = np.where(results['Original bin'] == bin_n)
            pred_spec = results.iloc[orig_bin_idx].loc[:,['Phot z', 'Spec z']].iterrows()
            # pred_CO_filtered = filter(lambda x: True if abs(x[1][0] - x[1][1]) > 1 else False, pred_spec)
            pred_CO_filtered = filter(lambda x: True if abs(x[1]['Phot z'] - x[1]['Spec z']) > 1 else False, pred_spec)
            # pred_CO_filtered = filter(lambda x: True if abs(x.iloc[1,0] - x.iloc[1,1]) > 1 else False, pred_spec)
            pred_CO_count = sum([1 for _ in pred_CO_filtered])
            frac_temp.append(round(pred_CO_count / orig_bin_idx[0].size * 100, 2))

            frac_lists.append(frac_temp)

        fig = plt.figure(figsize = (7,7))
        ax1 = plt.subplot(1, 1, 1)

        plotting.plotpzsz(results['Spec z'], results['Phot z'])

        results_for_each_bins = ''
        for n, frac_pair in enumerate(frac_lists):
            NO_frac, CO_frac = frac_pair # unpack the pair
            results_for_each_bins += f'{bins[n]: .5f}~{bins[n+1]: .5f}:        NO: {NO_frac:0>5.2f}% CO: {CO_frac:0>5.2f}%\n'
        fig.text(0, -0.5, results_for_each_bins)

        fig.savefig(self.output_pdf_path, format="pdf", bbox_inches="tight")

        # save CSV file
        if not full_save:
            # only save phot-z without 5-band magnitudes
            results = results.loc[:, ['Spec z', "Phot z"]]
        results.to_csv(Path(self.output_csv_path), index = False) # index = False gets rid of unnamed column 0

        return