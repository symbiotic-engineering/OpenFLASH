# syntax=docker/dockerfile:1
# Two stages: build the Conda environment with the full Conda/Mamba tooling,
# then copy only the finished environment onto a LaTeX image. The base Conda
# install, package cache, and compilers stay behind in the builder.
#
# The package source isn't copied in (see the .pth file below), so the image
# only needs rebuilding when the environment file or pyproject.toml changes.

ARG ENV_PREFIX=/opt/conda/envs/openflash.openflash-on-branch-tex

FROM condaforge/miniforge3:latest AS builder
ARG ENV_PREFIX

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY analysis/environment-branch.yml /tmp/analysis/environment-branch.yml

# Create the environment from the Conda packages only (strip the pip
# section, which just installs the local package).
RUN sed '/^[[:space:]]*- pip:/,$d' /tmp/analysis/environment-branch.yml \
      > /tmp/analysis/environment-nopip.yml \
 && mamba env create -p "$ENV_PREFIX" -f /tmp/analysis/environment-nopip.yml

# If libquadmath is missing (i.e., on aarch64), drop an empty stub into the
# env so the linker is satisfied when building capytaine from source, which
# references no quadmath symbols. Conda links extensions with an rpath to
# $CONDA_PREFIX/lib, so the stub resolves at both link and run time.
RUN mamba run -p "$ENV_PREFIX" bash -c '\
  if [ ! -e "$CONDA_PREFIX/lib/libquadmath.so" ]; then \
    echo "void __quadmath_stub(void){}" > /tmp/quadmath_stub.c \
    && "$CC" -shared -fPIC -Wl,-soname,libquadmath.so.0 \
         -o "$CONDA_PREFIX/lib/libquadmath.so.0" /tmp/quadmath_stub.c \
    && ln -sf libquadmath.so.0 "$CONDA_PREFIX/lib/libquadmath.so"; \
  fi'

# Install the local package's dependencies, but not the package itself, so
# the image depends on pyproject.toml alone and not the source code. The
# package is built from placeholder files, then uninstalled.
COPY pyproject.toml /tmp/openflash/pyproject.toml
RUN cd /tmp/openflash \
 && mkdir -p package/src/openflash \
 && touch README.md LICENSE CITATION.cff package/src/openflash/__init__.py \
 && mamba run -p "$ENV_PREFIX" python -m pip install . \
 && mamba run -p "$ENV_PREFIX" python -m pip uninstall -y open-flash

# Import the package straight from the project, which Calkit mounts at /work,
# so code changes take effect without rebuilding (and pushing) the image.
# This is what an editable install does, and a .pth file, unlike PYTHONPATH,
# survives commands that set their own PYTHONPATH
RUN echo /work/package/src > "$(mamba run -p "$ENV_PREFIX" python -c \
      'import sysconfig; print(sysconfig.get_path("purelib"))')/openflash-src.pth"

RUN mamba run -p "$ENV_PREFIX" python -m ipykernel install --prefix="$ENV_PREFIX"

# Drop the compilers (about 900 MB), which are only needed to build
# extensions. --force removes just these packages, keeping the runtime
# libraries (libgcc, libgfortran, etc.) that other packages need.
RUN <<EOF
set -eu
pkgs=$(mamba list -p "$ENV_PREFIX" --json | python -c '
import json, re, sys
pkgs = json.load(sys.stdin)
pkgs = pkgs["packages"] if isinstance(pkgs, dict) else pkgs
pattern = (
    r"(c|fortran)-compiler|(gcc|gfortran|binutils)(_impl)?(_linux-.*)?"
    r"|(sysroot|kernel-headers)_linux-.*|lib(gcc|stdcxx|gfortran)-devel_linux-.*"
    r"|libsanitizer|conda-gcc-specs"
)
print(" ".join(p["name"] for p in pkgs if re.fullmatch(pattern, p["name"])))
')
test -n "$pkgs"
mamba remove -p "$ENV_PREFIX" -y --force $pkgs
find "$ENV_PREFIX" -name '*.a' -delete
EOF

# Calkit's TinyTeX image, about 0.4 GB compressed versus 2.8 GB for TeX Live
# full. Pinned by digest so a rebuild doesn't pick up a new base, which would
# mean pushing every layer again
FROM ghcr.io/calkit/latex:0.1.7@sha256:1fd30a539762cecfca5be9f1e63201593012c08d1103e4155d51dc252d418aaf
ARG ENV_PREFIX

# What Matplotlib's text.usetex needs beyond the base image. dvipng is a
# compiled tool, so it can't be fetched on demand
RUN tlmgr update --self \
 && tlmgr install dvipng type1cm underscore \
 && find /opt/.TinyTeX ! -perm -a+r -exec chmod a+rX {} +

# Same path as in the builder, since Conda environments aren't relocatable
COPY --from=builder $ENV_PREFIX $ENV_PREFIX

ENV PATH=$ENV_PREFIX/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN useradd -m openflash \
 && mkdir -p /workspace \
 && chown -R openflash:openflash /workspace \
 && mkdir -p /home/openflash/.ipython /home/openflash/.cache /home/openflash/.config \
 && chown -R openflash:openflash /home/openflash \
 && mkdir -p /tmp/home/.local /tmp/home/.ipython /tmp/home/.cache /tmp/home/.config \
             /tmp/ipython /tmp/matplotlib /tmp/cache /tmp/jupyter \
 && chmod -R 777 /tmp/home /tmp/ipython /tmp/matplotlib /tmp/cache /tmp/jupyter

USER openflash
WORKDIR /workspace

ENV HOME=/tmp/home
ENV IPYTHONDIR=/tmp/home/.ipython
ENV XDG_CACHE_HOME=/tmp/home/.cache
ENV XDG_CONFIG_HOME=/tmp/home/.config
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV JUPYTER_DATA_DIR=/tmp/jupyter

CMD ["bash"]
