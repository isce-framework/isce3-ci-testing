ARG runtime_img
FROM $runtime_img

# Install dependencies for ISCE development
RUN set -ex \
 && yum update -y \
 && yum install -y \
        cuda-cudart-devel-$CUDA_PKG_VERSION \
        cuda-nvcc-$CUDA_PKG_VERSION \
        libcufft-devel-$CUDA_PKG_VERSION \
        doxygen \
        rpm-build \
 && yum clean all \
 && rm -rf /var/cache/yum \
 && rm -rf /var/cache/yum

COPY spec-file.txt /tmp/spec-file.txt
RUN conda install --yes --file /tmp/spec-file.txt \
 && conda clean -tipsy \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/spec-file.txt

ENV CUDAHOSTCXX=x86_64-conda_cos6-linux-gnu-g++
