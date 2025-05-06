#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import plotting_routine as plotting
from co_util import rebalance, CO_list, NCO_list


import json

class CO_BC:

    def __init__(self, **config):
        '''
        parameters

        '''
        # inputs
        self.input_csv_path_PDF = config.get("input_csv_path_PDF")
        self.input_PDF = self.input_csv_path_PDF.stem if self.input_csv_path_PDF else None
        
        self.input_csv_path_photz = config.get("input_csv_path_photz")
        self.input_photz = self.input_csv_path_photz.stem if self.input_csv_path_photz else None

        header = pd.read_csv(self.input_csv_path_PDF, nrows = 0) # include index_col = 0 if there is index

        self.evaluation = config.get("evaluation", True)
        if self.evaluation:
            self.evaluation_ratio = config["evaluation_ratio"]
            
        self.train_ratio = config.get("train_ratio")

        # hyperparametes
        self.num_input_features = len(header.columns) - 7 # dynamically determine num_input_features (7 includes 5 band magnitudes, spec z, phot z)
        self.num_hidden_neurons = config.get("num_hidden_neurons")
        self.num_hidden_layers = config.get("num_hidden_layers")
        self.num_epochs = config.get("num_epochs")
        self.learning_rate = config.get("learning_rate")
        self.size_batch = config.get("size_batch")
        self.weights = config.get("weights")
        self.CO_ratio = config.get("CO_ratio")
        self.weights = config.get("weights")
        # weights format:
        # [[w1_threshold, w2_ threshold, ...], [w1, w2, ...]]

        # output configurations
        self.model_no = config.get("model_no")

        self.suffix = f'{self.input_PDF}_{self.input_photz}_{int(config.get("CO_ratio") * 100)}_{self.weights}_{self.model_no}'
        
        if 'output_csv_path' not in config:
            self.output_csv_path = config.get("output_dir_path") / f"results_CO_BC_{self.suffix}.csv"
        else:
            self.output_csv_path = config.get("output_csv_path")
        
        if 'output_pdf_path' not in config:
            self.output_pdf_path = config.get("output_dir_path") / f"results_CO_BC_{self.suffix}.pdf"
        else:
            self.output_pdf_path = config.get("output_pdf_path")

        if 'model_path' not in config:
            self.model_path = config.get("model_dir_path") / f"results_CO_BC_{self.suffix}.pth"
        else:
            self.model_path = config.get("model_path")

        self.device = config.get("device")


    class NeuralNetwork(nn.Module):
        def __init__(self, num_input_features,
                     num_hidden_neurons: list,
                     num_hidden_layers):
            super(CO_BC.NeuralNetwork, self).__init__()
            self.input_layer = nn.Linear(num_input_features, num_hidden_neurons[0])
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(num_hidden_neurons[0], num_hidden_neurons[1]),
                 nn.Linear(num_hidden_neurons[1], num_hidden_neurons[2])]
            )
            self.output_layer = nn.Linear(num_hidden_neurons[2], 1)

        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            for hidden_layer in self.hidden_layers:
                x = torch.relu(hidden_layer(x))
            x = torch.sigmoid(self.output_layer(x))
            return x

    def preprocess_data(self):
        '''preprocess raw data to return df(training and evaluation dataset) and, if needed, eval_df(base evaluation dataset) as DataSet object
        input:
            evaluation is a bool value determining if evaluation is needed
        output:
            data is a dictionary containing,
                df is a DataFrame object for training and evaluation dataset
                rebalanced_df is df after rebalancing
                eval_df is a DataFrame object for base evaluation dataset (if needed)
        '''
        input_csv_path_PDF = self.input_csv_path_PDF
        raw = pd.read_csv(input_csv_path_PDF)

        # Get rid of 5-band flux columns
        raw_df = raw.iloc[:, 5:]

        input_csv_path_photz = self.input_csv_path_photz
        photz_df = pd.read_csv(input_csv_path_photz)
        raw_df.loc[:, 'Phot z'] = photz_df['Phot z']

        # Add CO column
        # CO: 1, NCO: 0
        spec_photo = pd.DataFrame(raw_df, columns = ['Spec z', 'Phot z'])

        raw_df = raw_df.copy()
        raw_df['CO?'] = 0
        raw_df.iloc[CO_list(raw_df).index, raw_df.columns.get_loc('CO?')] = 1
        raw_df = raw_df.dropna()

        CO_df = CO_list(raw_df)
        NCO_df = NCO_list(raw_df)
        num_balanced_NCO = int((1/self.CO_ratio - 1)*len(CO_df))

        balanced_df = rebalance(raw_df, self.CO_ratio)

        data = {"raw_df": raw_df, "balanced_df": balanced_df}
        if self.evaluation:
            # 3. Set aside evaluation set
            eval_df = raw_df.sample(frac = self.evaluation_ratio)
            df = raw_df.drop(eval_df.index).sample(frac = 1.0)
            data.update({"df": df, "eval_df": eval_df})

        else:
            df = raw_df.sample(frac = 1.0)

            data.update({"df": df})

        return data

    def build_dl(self, data):
        '''Build DataLoader objects for training and evaluations
        input:
            data is the output of preprocess_data()
        output:
            best_model

        '''
        df = data["df"]
        balanced_df = data["balanced_df"]
        eval_df = data["eval_df"]

        # 4. Splitting df into training data set and test data set
        # Split dataframe into z (Spec z), F (EPDF), T (CO?)
        z = balanced_df['Spec z']
        F = balanced_df.drop(['Spec z', 'Phot z', 'CO?'], axis = 1)
        T = balanced_df['CO?']

        idx_tensor = torch.IntTensor(F.index).to(self.device)
        z_tensor = torch.FloatTensor(z.values).to(self.device)
        F_tensor = torch.FloatTensor(F.values).to(self.device) # DataFrame.values extracts the data as a numpy array as tensor.Tensoris constructed from as an array
        T_tensor = torch.FloatTensor(T.values).to(self.device)

        # data for training and tests (and eval_data) for base
        tensorData = TensorDataset(idx_tensor, F_tensor, T_tensor, z_tensor)

        # Determine training and test sets
        train_size = int(len(tensorData) * self.train_ratio)
        test_size = len(tensorData) - train_size

        train_data, test_data = random_split(dataset = tensorData, lengths = [train_size, test_size])

        if self.evaluation:
            # 5. Splitting evaluation dataframe into inputs and outputs dataset
            eval_z = eval_df['Spec z']
            eval_F = eval_df.drop(['Spec z', 'Phot z', 'CO?'], axis = 1)
            eval_T = eval_df['CO?']

            eval_idx_tensor = torch.IntTensor(eval_F.index)
            eval_z_tensor = torch.FloatTensor(eval_z.values)
            eval_F_tensor = torch.FloatTensor(eval_F.values)
            eval_T_tensor = torch.FloatTensor(eval_T.values)

            eval_data = TensorDataset(eval_idx_tensor, eval_F_tensor, eval_T_tensor, eval_z_tensor)
            eval_data

        # Initialize DataLoader objects
        train_dl = DataLoader(dataset = train_data,
                              batch_size = self.size_batch,
                              shuffle = True)
        test_dl = DataLoader(dataset = test_data,
                              batch_size = self.size_batch,
                              shuffle = False)
        dl = {"train_dl": train_dl, "test_dl": test_dl}

        if self.evaluation:
            eval_dl = DataLoader(dataset = eval_data,
                                      batch_size = self.size_batch,
                                      shuffle = False)
            dl.update({"eval_dl": eval_dl})

        return dl

    def train_model(self, dl):

        # Initialize model
        model = CO_BC.NeuralNetwork(self.num_input_features, self.num_hidden_neurons, self.num_hidden_layers)
        model = model.to(self.device)

        # Define loss function and optimizer
        # Binary Cross Entropy
        criterion = nn.BCELoss()
        # Adaptive Moment Estimation
        optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)

        min_loss = 1
        best_model = None
        loss_data = []

        # Define training
        def train(model,inputs, targets, zs, optimizer, weights):
            model.train()

            inputs, targets, zs = inputs.to(self.device), targets.to(self.device), zs.to(self.device)
            
            # zs = np.asarray(zs).squeeze()
            outputs_pred = model(inputs).squeeze(dim = 1)

            # Determine weights
            weight_all_galaxies = torch.ones_like(targets, dtype = float)
            
            for z_threshold, weight in zip(*self.weights):
                z_threshold = torch.tensor([z_threshold]).to(self.device)
                weight = torch.tensor([weight], dtype = float).to(self.device)
                
                highreds = torch.where((zs > z_threshold) & (targets == 0))
                weight_all_galaxies[highreds] = weight

            # Define loss function with weights
            criterion = nn.BCELoss(weight = weight_all_galaxies)
            loss = criterion(outputs_pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss

        # Train the model
        num_epochs = self.num_epochs
        train_dl = dl["train_dl"]
        for epoch in range(num_epochs):
            # Train
            for i, inputs, targets, zs in train_dl:
                loss = train(model, inputs, targets, zs, optimizer, self.weights)

            # Record the minimum loss
            if loss < min_loss:
                min_loss = loss.item()
                best_model = model.state_dict()
                torch.save(best_model, self.model_path)
                print('best model')

            # Print
            if ((epoch + 1)%10 == 0) or (loss == min_loss):
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.12f}')
                loss_data.append(loss.item())
                print()

        return best_model

    def evaluate_model(self, data, dl):
        best_model = CO_BC.NeuralNetwork(self.num_input_features, self.num_hidden_neurons, self.num_hidden_layers)
        best_model = best_model.to(self.device)
        best_model.load_state_dict(torch.load(self.model_path))

        # Evaluate model performance
        # Set the module in evaluation mode -- some specific layers stop training
        best_model.eval()

        # Deactivates autograd engine -- reduces memory usage and speed up the computation
        with torch.no_grad():
            idx = []
            CO_true = []
            CO_pred = []

            # try:
            if self.evaluation:
                for i, inputs, targets, zs in dl["eval_dl"]:
                    
                    outputs = best_model(inputs)

                    idx.extend(i.numpy())
                    CO_true.extend(targets.numpy())
                    CO_pred.extend(outputs.numpy().squeeze(axis = 1))

            else:
                for i, inputs, targets, zs in dl["test_dl"]:
                    outputs = best_model(inputs)\

                    idx.extend(i.numpy())
                    CO_true.extend(targets.numpy())
                    CO_pred.extend(outputs.numpy().squeeze(axis = 1))

        # Evaluate the best model with the evaluation set
        idx = np.array(idx)
        CO_true = np.array(CO_true)
        CO_pred = np.array(CO_pred)

        if self.evaluation:
            eval_df = data["eval_df"]
            spec_z = eval_df.loc[idx, 'Spec z']
            phot_z = eval_df.loc[idx, 'Phot z']

        else:
            df = data["df"]
            spec_z = df.loc[idx, 'Spec z']
            phot_z = df.loc[idx, 'Phot z']

        results = pd.DataFrame({
            'CO?': CO_true,
            'Predicted CO?': CO_pred,
            'Spec z': spec_z,
            'Phot z': phot_z
        })

        return results

    def save_results(self, data, results, evaluation_ratio = 1.0):

        n_bins = 10

        CO_results = CO_list(results)['Predicted CO?']
        NCO_results = NCO_list(results)['Predicted CO?']

        fig = plt.figure(figsize = (7, 15.75))
        ax1 = plt.subplot(3, 1, 1)

        # Probability plot
        # weights = [np.ones_like(CO_results) / len(CO_results), np.ones_like(NCO_results) / len(NCO_results)]
        # values, bins, _ = plt.hist([CO_results, NCO_results], weights = weights, bins = n_bins)
        plt.legend(['CO', 'NCO'])

        # # Counts
        # plt.hist([CO_results, NCO_results], bins = n_bins)[2]
        # plt.legend(['CO', 'NCO'])

        # # Probability density plot
        # plt.hist([CO_results, NCO_results], density = True, bins = n_bins)[2]

        df = data["raw_df"]

        orig_phot_z = np.array(df['Phot z'])
        orig_spec_z = np.array(df['Spec z'])

        ax2 = plt.subplot(3, 1, 1)
        z_vs_num = plotting.plotN(orig_spec_z)

        ax3 = plt.subplot(3, 1, 2)
        specz_photz = plotting.plotpzsz(orig_spec_z, orig_phot_z, 0.3)

        temp = np.array([
            np.array(results['Phot z']),
            np.array(results['CO?']),
            np.array(results['Spec z']),
            np.array(results['Predicted CO?'])
        ])

        # plotting.plotvsz(temp.transpose(), 0.5)
        ax4 = plt.subplot(3, 1, 3)
        z_vs_fraction = plotting.plotvsz(temp.transpose(), 0.5, evaluation_ratio = evaluation_ratio)

        fig.tight_layout()
        fig.text(0, -0.1, z_vs_fraction)

        fig.savefig(self.output_pdf_path, format="pdf", bbox_inches="tight", pad_inches = 0)
        plt.close(fig)

        # results.set_index(idx, inplace = True) # Use inplace parameter to determine whether to modify the DF object or create a new one
        results.to_csv(self.output_csv_path, index = False)
        print(f"Saved results to {self.output_csv_path}")

        return