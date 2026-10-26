FROM rockylinux/rockylinux:10.1-minimal AS builder

# Refresh the OS trust store before downloading build tools or runtimes.
RUN microdnf update -y && microdnf install -y ca-certificates && \
    update-ca-trust && microdnf clean all
# Explicitly distrust the root withdrawn in certifi 2026.10.19.
COPY emsign-ecc-root-g3.pem /etc/pki/ca-trust/source/blocklist/emsign-ecc-root-g3.pem
RUN update-ca-trust

# Seed Hatch virtual environments from a verified patched pip wheel.
ADD --checksum=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70 https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl /opt/python-bootstrap/pip-26.2.2-py3-none-any.whl
ENV VIRTUALENV_PIP=26.2.2 \
    VIRTUALENV_EXTRA_SEARCH_DIR=/opt/python-bootstrap \
    VIRTUALENV_DOWNLOAD=false \
    UV_SYSTEM_CERTS=true \
    SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

RUN microdnf -y update && \
    microdnf -y install curl tar python3 python3-pip python3-setuptools gcc && \
    microdnf clean all

RUN curl -L https://github.com/pypa/hatch/releases/download/hatch-v1.14.0/hatch-x86_64-unknown-linux-gnu.tar.gz \
      -o /tmp/hatch.tar.gz && \
    tar -xzf /tmp/hatch.tar.gz -C /tmp && \
    install -m 0755 /tmp/hatch /usr/local/bin/hatch && \
    rm -rf /tmp/hatch* /tmp/hatch.tar.gz

WORKDIR /src
COPY . /src

# Build a wheel for your project (outputs into /src/dist)
RUN hatch build -t wheel

FROM rockylinux/rockylinux:10.1-minimal AS runtime

# Refresh the OS trust store before downloading build tools or runtimes.
RUN microdnf update -y && microdnf install -y ca-certificates && \
    update-ca-trust && microdnf clean all
# Explicitly distrust the root withdrawn in certifi 2026.10.19.
COPY emsign-ecc-root-g3.pem /etc/pki/ca-trust/source/blocklist/emsign-ecc-root-g3.pem
RUN update-ca-trust

# Seed Hatch virtual environments from a verified patched pip wheel.
ADD --checksum=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70 https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl /opt/python-bootstrap/pip-26.2.2-py3-none-any.whl
ENV VIRTUALENV_PIP=26.2.2 \
    VIRTUALENV_EXTRA_SEARCH_DIR=/opt/python-bootstrap \
    VIRTUALENV_DOWNLOAD=false \
    UV_SYSTEM_CERTS=true \
    SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt


RUN microdnf -y update && \
    microdnf -y install python3 python3-pip git && \
    microdnf clean all

# Non-root user
ENV HOME=/home/neo
RUN /usr/sbin/groupadd -g 1000 neo && \
    /usr/sbin/useradd -u 1000 -g 1000 -m -d ${HOME} -s /sbin/nologin neo && \
    mkdir -p /app && \
    chown -R 1000:1000 /app ${HOME}

USER neo
WORKDIR /app

# Create a venv and install only wheel
ENV VIRTUAL_ENV=/app/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}:/usr/bin"
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" && \
    /app/venv/bin/pip install --no-cache-dir --upgrade "certifi>=2026.10.19" && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip

COPY --from=builder /src/dist/*.whl /app/dist/

RUN /app/venv/bin/pip install --no-cache-dir /app/dist/*.whl && \
    ml4floods-cli --help
