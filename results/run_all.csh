#!/bin/csh

set echo

./run_setup.csh >! setup.txt
./run_amigos.csh
./run_symv.csh
./run_syev_2stage.csh
./run_syev.csh
./run_svd.csh
./run_geev.csh
