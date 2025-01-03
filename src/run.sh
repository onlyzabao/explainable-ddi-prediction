#!/usr/bin/env bash

file_path=$1
params=""
while IFS= read -r line || [ -n "$line" ]; do
  name="--$(cut -d '=' -f1 <<<"$line")"
  val="$(cut -d '=' -f2 <<<"$line")"
  val="$(cut -d '"' -f2 <<<"$val")"
  params="$params $name $val"
done <"$file_path"
export PYTHONPATH="."
gpu_id=1
cmd="python model/trainer.py $params"

$cmd
