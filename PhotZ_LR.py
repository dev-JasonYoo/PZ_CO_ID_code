import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import MinMaxScaler
import plotting_routine as plotting
import json

class PhotZ_LR:
  '''Neural network linear regression that predicts photometric red shift values of galaxies

  method:
  __init__
  preprocess_data
  build_dl
  train_model
  evaulate_model
  save_results

  class:
  NeuralNetwork
  '''

  def __init__(self, **config):
    self.num_input_features = config["num_input_features"]
    self.num_hidden_neurons = config["num_hidden_neurons"]
    self.num_hidden_layers = config["num_hidden_layers"]
    self.num_epochs = config["num_epochs"]
    self.learning_rate = config["learning_rate"]
    self.size_batch = config["size_batch"]
    self.momentum = config["momentum"]
    self.train_ratio = config["train_ratio"]

    self.model_no = config["model_no"]
    self.input_csv_path = config["input_csv_path"]
    
    self.suffix = f'PhotZ_LR_model_{self.input_csv_path.stem}_{self.model_no}.pth'
    if 'model_path' not in config:
        self.model_path = config.get("model_dir_path") / f"results_CO_BC_{self.suffix}.pth"
    else:
        self.model_path = config.get("model_path")

    if 'output_csv_path' not in config:
        self.output_csv_path = config.get("output_dir_path") / f"results_CO_BC_{self.suffix}.csv"
    else:
        self.output_csv_path = config.get("output_csv_path")
    
    if 'output_pdf_path' not in config:
        self.output_pdf_path = config.get("output_dir_path") / f"results_CO_BC_{self.suffix}.pdf"
    else:
        self.output_pdf_path = config.get("output_pdf_path")

    self.evaluation = config["evaluation"]
    if self.evaluation:
      self.evaluation_ratio = config["evaluation_ratio"]

  class NeuralNetwork(nn.Module):
    def __init__(self, num_input_features,
                num_hidden_neurons,
                num_hidden_layers):
      # Call __init__() of superclass (nn.Module)
      super(PhotZ_LR.NeuralNetwork, self).__init__()
      # nn.Linear returns Tensor object (vector, matrix, tensor)
      self.input_layer = nn.Linear(num_input_features, num_hidden_neurons)
      # nn.ModuleLIst?
      self.hidden_layers = nn.ModuleList([nn.Linear(num_hidden_neurons, num_hidden_neurons) for n in range(num_hidden_layers - 1)])
      self.output_layer = nn.Linear(num_hidden_neurons, 1)

    # Define the computation performed in every cell
    def forward(self, x):
      x = torch.relu(self.input_layer(x))
      for hidden_layer in self.hidden_layers:
        x = torch.relu(hidden_layer(x))
      x = torch.relu(self.output_layer(x))
      return x

  def preprocess_data(self):
    # raw_df = pd.read_csv(self.input_csv_path)
    raw_df = pd.read_csv(self.input_csv_path)
    scaler = MinMaxScaler()
    raw_df.iloc[:, :5] = pd.DataFrame(scaler.fit_transform(raw_df.iloc[:, :5]))

    raw_df = raw_df.iloc[:, :6]

    data = {"raw_df": raw_df}
    if self.evaluation:
      eval_df = raw_df.sample(frac = self.evaluation_ratio)
      df = raw_df.drop(eval_df.index).sample(frac = 1.0)
      data.update({"df": df, "eval_df": eval_df})

    else:
      df = raw_df.sample(frac = 1.0)

      data.update({"df": df})

    return data

  def build_dl(self, data):
    # raw_df = df + eval_df
    raw_df = data["raw_df"]
    df = data["df"]
    eval_df = data["eval_df"]

    # Split data into F and T for raw_df
    raw_F = raw_df.iloc[:, 0:5]
    raw_T = raw_df.iloc[:, 5]

    raw_idx_tensor = torch.IntTensor(raw_F.index)
    raw_F_tensor = torch.FloatTensor(raw_F.values)
    raw_T_tensor = torch.FloatTensor(raw_T.values)

    raw_data = TensorDataset(raw_idx_tensor, raw_F_tensor, raw_T_tensor)

    # Split data into F and T for data
    F = df.iloc[:, 0:5]
    T = df.iloc[:, 5]

    idx_tensor = torch.IntTensor(F.index)
    F_tensor = torch.FloatTensor(F.values)
    T_tensor = torch.FloatTensor(T.values)

    FT_data = TensorDataset(idx_tensor, F_tensor, T_tensor)

    # Determine training and test sets
    train_size = int(len(df) * self.train_ratio)
    test_size = len(df) - train_size

    train_data, test_data = random_split(dataset = FT_data,
                                        lengths = [train_size, test_size])

    if self.evaluation:
        # 5. Splitting evaluation dataframe into inputs and outputs dataset
        eval_F = eval_df.iloc[:, 0:5]
        eval_T = eval_df.iloc[:, 5]

        eval_idx_tensor = torch.IntTensor(eval_F.index)
        eval_F_tensor = torch.FloatTensor(eval_F.values)
        eval_T_tensor = torch.FloatTensor(eval_T.values)

        eval_data = TensorDataset(eval_idx_tensor, eval_F_tensor, eval_T_tensor)

    # Initialize DataLoader objects
    galaxy_dl = DataLoader(dataset = raw_data,
                      batch_size = self.size_batch,
                      shuffle = False)
    train_dl = DataLoader(dataset = train_data,
                          batch_size = self.size_batch,
                          shuffle = True)
    test_dl = DataLoader(dataset = test_data,
                          batch_size = self.size_batch,
                          shuffle = False)

    dl = {"galaxy_dl": galaxy_dl, "train_dl": train_dl, "test_dl": test_dl}
    if self.evaluation:
        eval_dl = DataLoader(dataset = eval_data,
                                  batch_size = self.size_batch,
                                  shuffle = False)
        dl.update({"eval_dl": eval_dl})

    return dl

  def train_model(self, dl):
    # Initialize model
    model = PhotZ_LR.NeuralNetwork(self.num_input_features,
                          self.num_hidden_neurons,
                          self.num_hidden_layers)

    # Define loss function and optimizer
    # Mean Sqaured Error loss function
    criterion = nn.MSELoss()
    # Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr = self.learning_rate, momentum = self.momentum)

    min_loss = 1 # why 1? what are the expected range of loss?
    best_model = None
    loss_data = []

    train_dl = dl["train_dl"]

    # Train the model
    num_epochs = self.num_epochs
    for epoch in range(num_epochs):
      # Train
      for idx, inputs, targets in train_dl:
        # set p.grad to 0 where p is a parameter
        optimizer.zero_grad()
        outputs_pred = model(inputs)
        loss = criterion(outputs_pred, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
    #     if epoch%10 == 0: print(inputs, targets.unsqueeze(1), outputs_pred)

      # Record the minimum loss
      if loss < min_loss:
        min_loss = loss.item()
        best_model = model.state_dict()
        torch.save(best_model, self.model_path / f'PhotZ_LR_model_{self.input_csv_path.stem}_{self.model_no}.pth')
        print('best model')

      # Print
      if ((epoch + 1)%10 == 0) or (loss == min_loss):
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_data.append(loss.item())
        print()

    return best_model

  def evaluate_model(self, data, dl):
    model_save_path = self.model_path

    best_model = PhotZ_LR.NeuralNetwork(self.num_input_features,
                              self.num_hidden_neurons,
                              self.num_hidden_layers)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.eval()

    with torch.no_grad():
        idx = []
        y_true = []
        y_pred = []

        for i, inputs, targets in dl["galaxy_dl"]:
            outputs = best_model(inputs)

            idx.extend(i.numpy())
            y_true.extend(targets.numpy())
            y_pred.extend(outputs.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    results_temp = pd.DataFrame({
        'idx': idx,
        'Spec z': y_true,
        'Phot z': y_pred.squeeze(),
        'mse': mse
    })

    results_temp.set_index('idx', inplace = True)
    results_temp.index.name = None
    results = pd.concat([data["raw_df"].iloc[:, :5], results_temp], axis = 1)

    return results

  def save_results(self, results):
    results.to_csv(self.output_csv_path, index = False)

    fig = plt.figure(figsize = (7,7))
    axes = plt.subplot(1,1,1)
    plotting.plotpzsz(results['Spec z'], results['Phot z'])

    fig.text(0, -0.1, results["mse"][0])

    fig.savefig(self.output_csv_path, format="pdf", bbox_inches="tight")
    
    return