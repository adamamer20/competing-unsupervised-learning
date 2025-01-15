FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN git clone https://github.com/adamamer20/competing-unsupervised-learning
WORKDIR /bio-unsupervised
RUN uv sync
ENTRYPOINT ["bash"]