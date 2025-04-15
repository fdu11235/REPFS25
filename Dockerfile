FROM continuumio/miniconda3

# Create environment with TA-Lib + GPU-compatible PyTorch + stable deps
COPY environment.yml .

RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yml && \
    conda clean -afy

# Activate conda environment
SHELL ["conda", "run", "-n", "trading-env", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app
COPY . .

CMD ["bash"]
