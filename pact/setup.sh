#!/bin/bash


if [ ! -d "lib/" ]
then
    mkdir lib
fi

if [ ! -d "lib/BalancedGo" ] 
then
    git clone --branch shellio https://github.com/lnz/BalancedGo.git lib/BalancedGo
    cd lib/BalancedGo
    go build
    cd ../..
fi

if [ ! -d "lib/nauty" ] 
then
    curl "https://pallini.di.uniroma1.it/nauty2_8_6.tar.gz" --output lib/nauty.tgz
    tar -xzf lib/nauty.tgz -C lib/
    rm lib/nauty.tgz
    cd lib
    mv nauty2_8_6 nauty
    cd nauty
    ./configure
    make
    cd ../..
fi

pipenv install
