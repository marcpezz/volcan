FROM quay.io/uninuvola/base:main

# DO NOT EDIT USER VALUE
USER root

## -- ADD YOUR CODE HERE !! -- ##
RUN conda create -n MTK -y  python=3.10 numpy scipy \
    pandas h5py tqdm matplotlib seaborn scikit-learn \
    umap-learn ipykernel && \
    conda clean -afy && \
    conda init && \
    /opt/conda/envs/MTK/bin/python -m ipykernel install --name  MTK --display-name MTK && \
    /opt/conda/bin/conda shell.bash deactivate

# Install Julia 1.10.4
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.5-linux-x86_64.tar.gz \
    && tar -xzf julia-1.12.5-linux-x86_64.tar.gz \
    && mv julia-1.12.5 /opt/julia-1.12 \
    && rm julia-1.12.5-linux-x86_64.tar.gz

# Add Julia to PATH
ENV PATH="/opt/julia-1.12/bin:$PATH"

# Verify Julia installation
RUN julia --version

COPY Project.toml Manifest.toml ./
#RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
RUN julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

## --------------------------- ##

RUN echo "/opt/conda/bin/conda init > /dev/null " >> /etc/profile.d/conda.sh && \
    echo "exec /bin/bash" >> /etc/profile

## --------------------------- ##

# DO NOT EDIT USER VALUE
USER jovyan
