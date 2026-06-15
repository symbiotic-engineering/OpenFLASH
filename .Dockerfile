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
    -n openflash-openflash-on-branch \
    -f /tmp/analysis/environment-branch.yml

ENV PATH=/opt/conda/envs/openflash-openflash-on-branch/bin:$PATH

WORKDIR /workspace

CMD ["bash"]