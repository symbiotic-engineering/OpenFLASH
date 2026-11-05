# syntax=docker/dockerfile:1
# Two stages: build the Conda environment with the full Conda/Mamba tooling,
# then copy only the finished environment onto a LaTeX image. The base Conda
# install, package cache, and compilers stay behind in the builder.
#
# The package source isn't copied in (see the .pth file below), so the image
# only needs rebuilding when the environment file or pyproject.toml changes.

ARG ENV_NAME=openflash-on-main
ARG ENV_PREFIX=/opt/conda/envs/openflash.${ENV_NAME}-tex

FROM condaforge/miniforge3:latest AS builder

ARG ENV_NAME
ARG ENV_PREFIX
ARG TARGETARCH

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY .calkit/env-locks/${ENV_NAME}/ /tmp/



RUN set -eux; \
    echo "TARGETARCH=$TARGETARCH"; \
    echo "ENV_PREFIX=$ENV_PREFIX"; \
    echo "Available lock files:"; \
    ls -lah /tmp/*.yml; \
    case "$TARGETARCH" in \
      amd64) cp /tmp/linux-64.yml /tmp/conda.lock ;; \
      arm64) cp /tmp/linux-aarch64.yml /tmp/conda.lock ;; \
      *) echo "Unsupported architecture: $TARGETARCH"; exit 1 ;; \
    esac; \
    echo "Selected lock file:"; \
    head -20 /tmp/conda.lock; \
    conda env create --solver classic -y -p "$ENV_PREFIX" -f /tmp/conda.lock

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

# normalize timestamps for reproducibility
ARG SOURCE_DATE_EPOCH=0
RUN find "$ENV_PREFIX" -depth \
    -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

# Calkit's TinyTeX image, about 0.4 GB compressed versus 2.8 GB for TeX Live
# full, which includes what Matplotlib's text.usetex needs. Pinned by digest
# so a rebuild doesn't pick up a new base, which would mean pushing every
# layer again
FROM ghcr.io/calkit/latex:0.1.8@sha256:711a840402dd7c5a06bb630d1d1ffd19595d42253939609b0c00dec1bf3762d3
ARG ENV_PREFIX

# Same path as in the builder, since Conda environments aren't relocatable
COPY --from=builder $ENV_PREFIX $ENV_PREFIX

ENV PATH=$ENV_PREFIX/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

ARG SOURCE_DATE_EPOCH=0

RUN groupadd --gid 1001 openflash \
 && useradd --uid 1001 --gid 1001 -m openflash \
 && mkdir -p /workspace \
 && chown -R openflash:openflash /workspace \
 && mkdir -p /home/openflash/.ipython /home/openflash/.cache /home/openflash/.config \
 && chown -R openflash:openflash /home/openflash \
 && mkdir -p /tmp/home/.local /tmp/home/.ipython /tmp/home/.cache /tmp/home/.config \
             /tmp/ipython /tmp/matplotlib /tmp/cache /tmp/jupyter \
 && chmod -R 777 /tmp/home /tmp/ipython /tmp/matplotlib /tmp/cache /tmp/jupyter \
 && find /etc/passwd /etc/group /etc/shadow /etc/gshadow \
         /workspace /home/openflash /tmp/home /tmp/ipython \
         /tmp/matplotlib /tmp/cache /tmp/jupyter \
      -depth -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

USER openflash
WORKDIR /workspace

ENV HOME=/tmp/home
ENV IPYTHONDIR=/tmp/home/.ipython
ENV XDG_CACHE_HOME=/tmp/home/.cache
ENV XDG_CONFIG_HOME=/tmp/home/.config
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV JUPYTER_DATA_DIR=/tmp/jupyter

CMD ["bash"]
