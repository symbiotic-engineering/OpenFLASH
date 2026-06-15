FROM condaforge/miniforge3:latest AS miniconda

FROM texlive/texlive:latest-full

COPY --from=miniconda /opt/conda /opt/conda

ENV PATH=/opt/conda/bin:$PATH

COPY analysis/environment-branch.yml /tmp/analysis/environment-branch.yml
COPY pyproject.toml /tmp/pyproject.toml
COPY package/src /tmp/package/src
COPY README.md /tmp/README.md
COPY LICENSE /tmp/LICENSE
COPY CITATION.cff /tmp/CITATION.cff

RUN mamba env create \
    -n openflash.openflash-on-branch-tex \
    -f /tmp/analysis/environment-branch.yml

ENV PATH=/opt/conda/envs/openflash.openflash-on-branch-tex/bin:$PATH

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN useradd -m openflash

RUN mkdir -p /workspace  \
 && chown -R openflash:openflash /workspace

RUN mkdir -p /home/openflash/.ipython /home/openflash/.cache /home/openflash/.config \
 && chown -R openflash:openflash /home/openflash

RUN mkdir -p /tmp/home/.local /tmp/home/.ipython /tmp/home/.cache /tmp/home/.config \
             /tmp/ipython /tmp/matplotlib /tmp/cache /tmp/jupyter \
 && chmod -R 777 /tmp/home /tmp/ipython /tmp/matplotlib /tmp/cache  /tmp/jupyter

RUN python -m ipykernel install \
      --prefix=/opt/conda/envs/openflash.openflash-on-branch-tex

USER openflash
WORKDIR /workspace

ENV HOME=/tmp/home
ENV IPYTHONDIR=/tmp/home/.ipython
ENV XDG_CACHE_HOME=/tmp/home/.cache
ENV XDG_CONFIG_HOME=/tmp/home/.config
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV JUPYTER_DATA_DIR=/tmp/jupyter

CMD ["bash"]