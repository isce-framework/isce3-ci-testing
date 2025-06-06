# Use pinned oraclelinux:8 image
ARG repository=cae-artifactory.jpl.nasa.gov:16003/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux
ARG tag=8.10.250401
ARG digest=sha256:e37c210f26cfe7660808e536937fcf6ac359b3ec5713211e46519d7a35973a57
FROM ${repository}:${tag}@${digest}

RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Trying to install a package that doesn't exist should be an error.
RUN yum update -y \
 && yum clean all \
 && echo 'skip_missing_names_on_install=False' >> /etc/yum.conf \
 && rm -rf /var/cache/yum

# install latest miniforge
ARG conda_prefix
RUN curl -sSL https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-Linux-x86_64.sh \
        -o Miniforge3.sh \
 && bash Miniforge3.sh -b -p $conda_prefix \
 && rm Miniforge3.sh
ENV PATH="$conda_prefix/bin:$PATH"

# use the libmamba solver, which is faster and has better diagnostics
RUN conda config --set solver libmamba

COPY spec-file.txt /tmp/spec-file.txt
RUN conda install --yes --file /tmp/spec-file.txt \
 && conda clean -ay \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/spec-file.txt

# set up conda environment
RUN echo ". $conda_prefix/etc/profile.d/conda.sh" >> /etc/bashrc \
 && echo "conda activate base"                    >> /etc/bashrc
ENV GDAL_DATA=$conda_prefix/share/gdal
ENV GDAL_DRIVER_PATH=$conda_prefix/lib/gdalplugins
ENV PROJ_LIB=$conda_prefix/share/proj
ENV MPLCONFIGDIR=/tmp

# In GDAL>=3.10, constructing a "MEM" raster from a pointer address is disallowed by
# default for security reasons. See https://gdal.org/en/stable/drivers/raster/mem.html.
ENV GDAL_MEM_ENABLE_OPEN=YES

ENV CUDA_VERSION_MAJOR 11
ENV CUDA_VERSION_MINOR 8
ENV CUDA_PKG_VERSION "${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"

RUN yum install -y \
        cuda-cudart-$CUDA_PKG_VERSION \
 && yum clean all \
 && rm -rf /var/cache/yum

# The libcufft-11-8 RPM is missing SHA256 digests, so the usual `yum install` command
# fails on some machines with 'package <...> does not verify: no digest'. See
# https://forums.developer.nvidia.com/t/unable-to-install-cuda-rpms-and-libraries-in-rhel8-fips-mode-no-digest/230008
# for details. Yum doesn't appear to support an equivalent to
# `rpm --install --nodigest`, so we need this roundabout approach of using yum to
# download the RPM(s) and then using rpm to install the package(s). This seems to be
# resolved in CUDA 12, so we could revert this when we upgrade from CUDA 11.
RUN yum install -y --downloadonly --destdir /tmp \
        libcufft-$CUDA_PKG_VERSION \
 && rpm --install --nodigest /tmp/*.rpm \
 && rm /tmp/*.rpm

# https://github.com/NVIDIA/nvidia-container-runtime#environment-variables-oci-spec
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Specify the minimum driver version required on the host. The CUDA runtime libraries
# installed within the container are backwards compatible with older drivers within the
# same major version family. However, a newer driver may still be required if the GPU
# architecture is different from the architectures that we target when compiling ISCE3
# within the container image.
ENV NVIDIA_REQUIRE_CUDA "cuda>=${CUDA_VERSION_MAJOR}"
