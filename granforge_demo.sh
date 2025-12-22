#!/bin/bash

if [ ! -f build/CMakeCache.txt ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build --config Release
#check if examples/data/job/parquet contains parquet files
if [ ! -d "examples/data/job/parquet" ] || [ -z "$(ls -A examples/data/job/parquet/*.parquet 2>/dev/null)" ]; then
  echo "Parquet files not found in examples/data/job/parquet. Downloading the files..."
  doi="https://doi.org/10.5281/zenodo.17803141"
  data_suffix="${doi##*/}"
  data_id="${data_suffix#zenodo.}"
  data_url="https://zenodo.org/api/records/${data_id}/files-archive"
  curl -L -o examples/data/job/parquet/job_data.zip "${data_url}"
  unzip examples/data/job/parquet/job_data.zip -d examples/data/job/parquet/
  unzip examples/data/job/parquet/parquet.zip -d examples/data/job/parquet/
  rm examples/data/job/parquet/job_data.zip
  rm examples/data/job/parquet/parquet.zip
fi
python3 examples/granforge_demo.py --schema-file=examples/data/job/parquet/schema.sql --data examples/data/job/parquet/complete_cast.parquet examples/data/job/parquet/title.parquet --sql-file=examples/demo.sql --backend=build/src/examples/granforge_demo

