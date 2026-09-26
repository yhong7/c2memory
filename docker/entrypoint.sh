#!/bin/sh
set -eu

mkdir -p /app/.db /app/data /app/logs /app/result /app/temp

exec "$@"
