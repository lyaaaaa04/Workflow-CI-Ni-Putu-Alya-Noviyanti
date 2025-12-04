FROM continuumio/miniconda3

# Install MLflow
RUN conda install -y -c conda-forge mlflow && conda clean -afy

# Copy conda environment file
COPY conda.yaml /tmp/conda.yaml

# Create environment
RUN conda env create -f /tmp/conda.yaml

# Ensure environment name matches your conda.yaml
ENV PATH /opt/conda/envs/mlflow-env/bin:$PATH

# Activate environment automatically
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Expose MLflow UI port
EXPOSE 5000

# Start MLflow server + run model training script
CMD mlflow server --host 0.0.0.0 --port 5000 & python modelling.py
