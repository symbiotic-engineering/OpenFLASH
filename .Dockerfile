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

# Build the Conda environment in three phases to work around the absence of
# libquadmath on aarch64: no native libquadmath package exists for arm64, yet
# gfortran still emits -lquadmath, which breaks capytaine's from-source build
# (capytaine ships no arm64 wheel).
#
# 1. Create the environment with the Conda packages only (strip the pip
#    section), which installs the compiler toolchain.
RUN sed '/^[[:space:]]*- pip:/,$d' /tmp/analysis/environment-branch.yml \
      > /tmp/analysis/environment-nopip.yml \
 && mamba env create \
      -n openflash.openflash-on-branch-tex \
      -f /tmp/analysis/environment-nopip.yml

# 2. If libquadmath is missing (i.e. on aarch64), drop an empty stub into the
#    env so the linker is satisfied. capytaine references no quadmath symbols,
#    and Conda links extensions with an rpath to $CONDA_PREFIX/lib, so the stub
#    resolves at both link and run time.
RUN mamba run -n openflash.openflash-on-branch-tex bash -c '\
  if [ ! -e "$CONDA_PREFIX/lib/libquadmath.so" ]; then \
    echo "void __quadmath_stub(void){}" > /tmp/quadmath_stub.c \
    && "$CC" -shared -fPIC -Wl,-soname,libquadmath.so.0 \
         -o "$CONDA_PREFIX/lib/libquadmath.so.0" /tmp/quadmath_stub.c \
    && ln -sf libquadmath.so.0 "$CONDA_PREFIX/lib/libquadmath.so" \
    && rm /tmp/quadmath_stub.c; \
  fi'

# 3. Install the pip packages; capytaine compiles its Fortran extension against
#    the stub.
RUN mamba env update \
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
