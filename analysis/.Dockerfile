FROM continuumio/miniconda3 AS miniconda

FROM texlive/texlive:latest-full

COPY --from=miniconda /opt/conda /opt/conda

ENV PATH=/opt/conda/bin:$PATH

COPY analysis/environment-branch.yml /tmp/build_env.yaml

RUN conda env create \
    -n openflash-openflash-on-branch \
    -f /tmp/build_env.yaml

ENV PATH=/opt/conda/envs/openflash-openflash-on-branch/bin:$PATH

WORKDIR /workspace

CMD ["bash"]