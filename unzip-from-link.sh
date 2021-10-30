#!/bin/bash

mkdir -p datasets

cd $2
curl -sS $1 > file.zip
unzip file.zip
rm file.zip