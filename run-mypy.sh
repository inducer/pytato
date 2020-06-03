#! /bin/bash

export MYPYPATH=$MYPYPATH:pytato/stubs/
mypy --strict pytato
