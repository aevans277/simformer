from sbibm import get_task as _get_torch_task

import torch
import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd

from scoresbibm.tasks.base_task import InferenceTask

def load_and_prepare_data(params_file, output_file):
    params_df = pd.read_csv(params_file)
    df = pd.read_csv(output_file)

    # filter data based on conditions
    df = df[(df['X'] <= 0.01) & (df['X'] >= -0.01) & (df['Y'] <= 0.01) & (df['Y'] >= -0.01)]
    df['X'] = df['X'] * 1000  # convert to mm
    df['Y'] = df['Y'] * 1000
    df['Vz'] = np.log1p(df['Vz'])  # log transformation using log1p to handle small values

    df = df.merge(params_df, on='simulation')

    # identify and remove rows with NaNs and infinite values
    columns = ['X', 'Y', 'Vx', 'Vy', 'Vz']
    conditioning_columns = ['cooling_beam_detuning', 'cooling_beam_radius', 'cooling_beam_power_mw',
                            'push_beam_detuning', 'push_beam_radius', 'push_beam_power',
                            'push_beam_offset', 'quadrupole_gradient', 'vertical_bias_field']

    initial_row_count = len(df)

    for col in columns + conditioning_columns:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"Column '{col}': {nan_count} NaNs, {inf_count} infinite values found removed.")
            df = df.dropna(subset=[col])
            df = df[~np.isinf(df[col])]

    removed_row_count = initial_row_count - len(df)
    print(f"Total rows removed: {removed_row_count}")

    normalisation_params = {}

    # normalise data columns
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / std
        normalisation_params[col] = (mean, std)

    # normalise conditioning columns
    for col in conditioning_columns:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / std
        normalisation_params[col] = (mean, std)

    data_to_learn = df[columns].values
    conditioning_data = df[conditioning_columns].values

    print("Imported experimental data.")

    return data_to_learn, conditioning_data, normalisation_params

class SBIBMTask(InferenceTask):
    observations = range(1, 11)

    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        self.task = _get_torch_task(self.name)

    def get_theta_dim(self):
        return self.task.dim_parameters

    def get_x_dim(self):
        return self.task.dim_data

    def get_prior(self):
        if self.backend == "torch":
            return self.task.get_prior_dist()
        else:
            raise NotImplementedError()

    def get_simulator(self):
        print('get_simulator')
        if self.backend == "torch":
            return self.task.get_simulator()
        else:
            raise NotImplementedError()

    def get_node_id(self):
        print('get_node_id')
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":
            return torch.arange(dim)
        else:
            return jnp.arange(dim)

    def get_data(self, num_samples: int, **kwargs):
        params_file = 'data/input/input_data(full).csv'
        output_file = 'data/input/compiled_vectors-0.05.csv'
        data_to_learn, conditioning_data, normalisation_params = load_and_prepare_data(params_file, output_file)

        print(f'x data type: {type(data_to_learn)}, shape: {data_to_learn.shape}')

        # Select a subset of the data if necessary
        if num_samples > data_to_learn.shape[0]:
            raise ValueError("num_samples exceeds the available data size.")

        indices = np.random.choice(data_to_learn.shape[0], num_samples, replace=False)

        # Use the real data from the data files
        if self.backend == "torch":
            thetas = torch.tensor(conditioning_data[indices, :9], dtype=torch.float16)  # Use float16
            xs = torch.tensor(data_to_learn[indices, :5], dtype=torch.float16)  # Use float16
            print('loaded torch data')
        elif self.backend == "jax":
            thetas = jnp.array(conditioning_data[indices, :9], dtype=jnp.float16)  # Use float16
            xs = jnp.array(data_to_learn[indices, :5], dtype=jnp.float16)  # Use float16
            print('loaded jax data')
        else:
            thetas = conditioning_data[indices, :9].astype(np.float16)  # Use float16
            xs = data_to_learn[indices, :5].astype(np.float16)  # Use float16
            print('loaded numpy data')

        print(f'theta type: {type(thetas)}, shape: {thetas.shape}')
        print(f'x type: {type(xs)}, shape: {xs.shape}')

        return {"theta": thetas, "x": xs}

    def get_observation(self, index: int):
        if self.backend == "torch":
            return self.task.get_observation(index)
        else:
            out = self.task.get_observation(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_reference_posterior_samples(self, index: int):
        if self.backend == "torch":
            return self.task.get_reference_posterior_samples(index)
        else:
            out = self.task.get_reference_posterior_samples(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_true_parameters(self, index: int):
        if self.backend == "torch":
            return self.task.get_true_parameters(index)
        else:
            out = self.task.get_true_parameters(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

class Mot(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="mot", backend=backend)

    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]])
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn