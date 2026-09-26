#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_root="${C2MEM_DATA_DIR:-$repo_root/data}"

mkdir -p "$data_root/fever" "$data_root/hotpotqa" "$data_root/wikipedia"

download_file() {
  local url=$1
  local destination=$2
  local partial="${destination}.part"

  if [[ -s "$destination" ]]; then
    printf 'Using existing file: %s\n' "$destination"
    return
  fi

  printf 'Downloading %s\n' "$destination"
  curl --fail --location \
    --retry 3 --retry-delay 2 --connect-timeout 30 \
    --output "$partial" "$url"
  mv "$partial" "$destination"
}

sha256_digest() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

md5_digest() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    md5 -q "$1"
  fi
}

check_digest() {
  local algorithm=$1
  local expected=$2
  local file=$3
  local actual

  if [[ "$algorithm" == "sha256" ]]; then
    actual=$(sha256_digest "$file")
  else
    actual=$(md5_digest "$file")
  fi

  if [[ "$actual" != "$expected" ]]; then
    printf 'Checksum mismatch for %s\nExpected: %s\nActual:   %s\n' \
      "$file" "$expected" "$actual" >&2
    return 1
  fi
  printf 'Verified %s\n' "$file"
}

fever_file="$data_root/fever/fever_dev.jsonl"
hotpotqa_file="$data_root/hotpotqa/hotpot_dev_distractor_v1.json"
wikipedia_file="$data_root/wikipedia/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"

download_file \
  'https://fever.ai/download/fever/paper_dev.jsonl' \
  "$fever_file"
check_digest sha256 \
  '41158707810008747946bf23471e82df53e77a513524b9e3ec1c2e674ef5ef8c' \
  "$fever_file"

# This mirror is byte-identical to the official HotpotQA distractor dev file.
download_file \
  'https://huggingface.co/datasets/RAGLAB/data/resolve/main/eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json?download=true' \
  "$hotpotqa_file"
check_digest sha256 \
  '4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea' \
  "$hotpotqa_file"

download_file \
  'https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2' \
  "$wikipedia_file"
check_digest md5 \
  '01edf64cd120ecc03a2745352779514c' \
  "$wikipedia_file"

printf 'Dataset preparation complete: %s\n' "$data_root"
