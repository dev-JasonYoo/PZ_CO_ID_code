from numpy import argmax, zeros, where, concatenate, digitize, asarray
from pandas.core.frame import DataFrame
from pandas.core.reshape.concat import concat

# Function returns 1 when CO
def determine_CO(z_phot, z_spec):
    if abs(z_phot - z_spec) > 1.0:
        return 1
    return 0

# Function returns 1 when O
def determine_O(z_phot, z_spec):
    if abs(z_phot - z_spec)/(1+z_spec) > 0.15:
        return 1
    return 0

def CO_list(galaxy_df: DataFrame):
    return galaxy_df[abs(galaxy_df["Phot z"] - galaxy_df["Spec z"]) > 1]

def NCO_list(galaxy_df: DataFrame):
    return galaxy_df[abs(galaxy_df["Phot z"] - galaxy_df["Spec z"]) <= 1]

def O_list(galaxy_df: DataFrame):
    return galaxy_df[abs(galaxy_df["Phot z"] - galaxy_df["Spec z"]) / (1 + galaxy_df["Spec z"]) > 0.15]

def NO_list(galaxy_df: DataFrame):
    return galaxy_df[abs(galaxy_df["Phot z"] - galaxy_df["Spec z"]) / (1 + galaxy_df["Spec z"]) <= 0.15]

def bin_no(y_pred_row: DataFrame):
    return int(argmax(y_pred_row))

def convert_bin_to_z(bin_no, bins):
    return (bins[bin_no] + bins[bin_no+1])/2

def assign_CO_flag(galaxy_df: DataFrame):
    CO_idx_list = CO_list(galaxy_df).index

    galaxy_df["CO?"] = zeros(len(galaxy_df))
    galaxy_df.loc[CO_idx_list, "CO?"] = 1

    return galaxy_df

def rebalance(galaxy_df: DataFrame, CO_ratio: float):

    CO_df = CO_list(galaxy_df)
    NCO_df = NCO_list(galaxy_df)
    num_balanced_NCO = int((1/CO_ratio - 1)*len(CO_df))
    
    balanced_NCO_df = NCO_df.sample(n = num_balanced_NCO)
    balanced_df = concat([CO_df, balanced_NCO_df])
    
    return balanced_df

def generate_bin(upper_limit):
    bins = [0]
    new_bin = 0
    lower_limit = 0
    while lower_limit < upper_limit:
        new_bin = round(new_bin + (1 + new_bin) * 0.15, 5) # rel-z dataset has up to 4 decimals, so 4+1 decimals for bins
        bins.append(new_bin)
        lower_limit = new_bin
    
    return bins

def rebalance_bins(galaxy_df: DataFrame, bin_no, rebalance_ratio): # DO NOT NAME PARAMETER AS 'df' -- it calls 'df' the global variable in for NumPy methods
    '''returns rebalanced dataset with given bin numbers and rebalancing ratios for each bin
    galaxy_df: pandas.DataFrame object that contains dataset (needs 'bin' column)
    bin_no: List of bin numbers of which bin needs rebalancing
    rebalance_ratio: List of ratios to rebalance for each bin in bin_no
    '''
    bin_n_idx_list = []
    balanced_bin_n_list = []
    
    for bin_n, ratio in zip(bin_no, rebalance_ratio):
        bin_n_idx = where(galaxy_df['bin no'] == bin_n)[0]
        bin_n_idx_list.append(bin_n_idx)
        balanced_bin_n_list.append(galaxy_df.iloc[bin_n_idx].sample(frac = ratio)) # iloc recieves and returns in 'integer positions'
    
    drop_list_idx = concatenate(bin_n_idx_list, axis = 0)
    drop_list_label = [galaxy_df.index[i] for i in drop_list_idx]
    rest = galaxy_df.drop(drop_list_label) # drop accesses rows by labels (NOT integer position)

    return concat([rest, *balanced_bin_n_list], axis = 0)

def assign_bin_no(galaxy_df: DataFrame, bins):
    galaxy_df['bin no'] = digitize(galaxy_df['Spec z'].to_numpy(), bins) # it puts 1 for bin 0
    galaxy_df['bin no'] -= 1

    return galaxy_df

def assign_one_hot_encoded_bin_no(galaxy_df: DataFrame, bins):
    num_bins = len(bins) - 1
    format_one_hot_encode = lambda x: [1 if i == x else 0 for i in range(num_bins)] # ex: if bin #3, [0, 0, 0, 1, 0, 0, ... ]
    galaxy_df['bin'] = asarray(map(format_one_hot_encode, galaxy_df['bin no']))
    
    return galaxy_df