FROM ubuntu:24.04

ENV DUCKDB_HOME=/usr/local/duckdb

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential sudo gdb \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3.12 \
    python3-pip \
    python3.12-venv \
    openjdk-17-jdk \
    g++ \
    ninja-build \
    libssl-dev \
    libboost-all-dev \
    libcurl4-gnutls-dev \
    libspdlog-dev && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip
    
RUN git clone https://github.com/duckdb/duckdb /tmp/duckdb && \
    mkdir -p /tmp/duckdb/build && \
    cd /tmp/duckdb/build && \
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${DUCKDB_HOME} && \
    ninja install

RUN mkdir -p /tmp/arrow && \
    git clone --branch apache-arrow-22.0.0 https://github.com/apache/arrow.git /tmp/arrow && \
    cd /tmp/arrow && \
    cmake -S ./cpp -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DARROW_BUILD_STATIC=ON \
        -DARROW_BUILD_SHARED=ON \
        -DARROW_COMPUTE=ON \
        -DARROW_CSV=ON \
        -DARROW_FILESYSTEM=ON \
        -DARROW_JEMALLOC=ON \
        -DARROW_JSON=ON \
        -DARROW_PARQUET=ON \
        -DARROW_S3=OFF \
        -DARROW_SUBSTRAIT=ON \
        -DARROW_WITH_LZ4=ON \
        -DARROW_WITH_BZ2=ON \
        -DARROW_WITH_SNAPPY=ON \
        -DARROW_WITH_ZLIB=ON \
        -DARROW_WITH_ZSTD=ON \
        -DARROW_HDFS=OFF \
        -DARROW_IPC=OFF \
        -DARROW_ORC=OFF \
        -DARROW_WITH_GRPC=BUNDLED \
        -DProtobuf_SOURCE=BUNDLED \
        -DgRPC_SOURCE=BUNDLED \
        -DARROW_FLIGHT=ON \
        -DARROW_FLIGHT_SQL=ON  && \
    cmake --build build && \
    cmake --install build

RUN mkdir -p /tmp/duckdb-substrait /tmp/substrait && \
    git clone --recursive https://github.com/substrait-io/duckdb-substrait-extension.git /tmp/duckdb-substrait && \
    cd /tmp/duckdb-substrait && \
    git reset --hard 36ce34be17b1ffe60810f98104b404870acdf682 && \
    make && \
    mv build/release /opt/duckdb_substrait_extension
    # get substrait version
    # source /tmp/arrow/cpp/thirdparty/versions.txt && \
    # git clone https://github.com/substrait-io/substrait.git /tmp/substrait && \
    # cd /tmp/substrait && \
    # git fetch --tags && \
    # SUBSTRAIT_VERSION_HASH=$(git rev-parse ${ARROW_SUBSTRAIT_BUILD_VERSION}) && \
    # cd /tmp/duckdb-substrait && \
    # sed -i "s|^GITHUB_TAG.*|GITHUB_TAG = \"${SUBSTRAIT_VERSION_HASH}\"|" scripts/update_substrait.py && \
    # python3 scripts/update_substrait.py && \
    

RUN /opt/venv/bin/pip install --no-cache-dir prompt_toolkit duckdb==1.3.0 substrait && \
    /opt/venv/bin/python -c "import duckdb; con = duckdb.connect(); con.execute(\"INSTALL '/opt/duckdb_substrait_extension/extension/substrait/substrait.duckdb_extension'\")" && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y protobuf-compiler libprotobuf-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    doxygen doxygen-doc doxygen-gui graphviz

RUN rm -rf /tmp/duckdb && rm -rf /tmp/arrow && rm -rf /tmp/proto && rm -rf /tmp/duckdb-substrait && rm -rf /var/lib/apt/lists/*


ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="/opt/venv/bin:$JAVA_HOME/bin/:${DUCKDB_HOME}/bin:$PATH"

WORKDIR /workspace
CMD ["bash", "/workspace/granforge_demo.sh"]