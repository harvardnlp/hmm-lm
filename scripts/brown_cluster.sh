#!/bin/bash

input=.data/penn-treebank/ptb.train.txt
finput=.data/penn-treebank/ptb.train.txt.flat

winput=.data/wikitext-103/wikitext-103/wiki.train.tokens
w2input=.data/wikitext-2/wikitext-2/wiki.train.tokens
w2finput=.data/wikitext-2/wikitext-2/wiki.train.tokens.flat
w2fainput=.data/wikitext-2/wikitext-2/wiki.train.tokens.flatarticles

wsjinput=.data/PTB/ptb.txt

supwsjinput=.data/PTB/sup/ptb.train.txt

if  [[ "$1" == "lm256" ]]; then
    $l_cluster --text $input --c 256 --output_dir clusters/lm-256
elif [[ "$1" == "lm128" ]]; then
    $l_cluster --text $input --c 128 --output_dir clusters/lm-128
elif [[ "$1" == "lm64" ]]; then
    $l_cluster --text $input --c 64 --output_dir clusters/lm-64
elif [[ "$1" == "lm32" ]]; then
    $l_cluster --text $input --c 32 --output_dir clusters/lm-32
elif [[ "$1" == "lm16" ]]; then
    $l_cluster --text $input --c 16 --output_dir clusters/lm-16
elif [[ "$1" == "lm10" ]]; then
    $l_cluster --text $input --c 10 --output_dir clusters/lm-10
elif [[ "$1" == "lm9" ]]; then
    $l_cluster --text $input --c 9 --output_dir clusters/lm-9
elif [[ "$1" == "lm8" ]]; then
    $l_cluster --text $input --c 8 --output_dir clusters/lm-8
elif [[ "$1" == "lm4" ]]; then
    $l_cluster --text $input --c 4 --output_dir clusters/lm-4

elif [[ "$1" == "flm256" ]]; then
    $l_cluster --text $finput --c 256 --output_dir clusters/flm-256
elif [[ "$1" == "flm128" ]]; then
    $l_cluster --text $finput --c 128 --output_dir clusters/flm-128
elif [[ "$1" == "flm64" ]]; then
    $l_cluster --text $finput --c 64 --output_dir clusters/flm-64
elif [[ "$1" == "flm32" ]]; then
    $l_cluster --text $finput --c 32 --output_dir clusters/flm-32
elif [[ "$1" == "flm16" ]]; then
    $l_cluster --text $finput --c 16 --output_dir clusters/flm-16
elif [[ "$1" == "flm10" ]]; then
    $l_cluster --text $finput --c 10 --output_dir clusters/flm-10
elif [[ "$1" == "flm9" ]]; then
    $l_cluster --text $finput --c 9 --output_dir clusters/flm-9
elif [[ "$1" == "flm8" ]]; then
    $l_cluster --text $finput --c 8 --output_dir clusters/flm-8
elif [[ "$1" == "flm4" ]]; then
    $l_cluster --text $finput --c 4 --output_dir clusters/flm-4

elif [[ "$1" == "wlm256" ]]; then
    $l_cluster --text $winput --c 256 --output_dir clusters/wlm-256
elif [[ "$1" == "wlm128" ]]; then
    $l_cluster --text $winput --c 128 --output_dir clusters/wlm-128
elif [[ "$1" == "wlm64" ]]; then
    $l_cluster --text $winput --c 64 --output_dir clusters/wlm-64
elif [[ "$1" == "wlm32" ]]; then
    $l_cluster --text $winput --c 32 --output_dir clusters/wlm-32
elif [[ "$1" == "wlm16" ]]; then
    $l_cluster --text $winput --c 16 --output_dir clusters/wlm-16
elif [[ "$1" == "wlm10" ]]; then
    $l_cluster --text $winput --c 10 --output_dir clusters/wlm-10
elif [[ "$1" == "wlm9" ]]; then
    $l_cluster --text $winput --c 9 --output_dir clusters/wlm-9
elif [[ "$1" == "wlm8" ]]; then
    $l_cluster --text $winput --c 8 --output_dir clusters/wlm-8
elif [[ "$1" == "wlm4" ]]; then
    $l_cluster --text $winput --c 4 --output_dir clusters/wlm-4

elif [[ "$1" == "w2lm256" ]]; then
    $l_cluster --text $w2input --c 256 --output_dir clusters/w2lm-256
elif [[ "$1" == "w2lm128" ]]; then
    $l_cluster --text $w2input --c 128 --output_dir clusters/w2lm-128
elif [[ "$1" == "w2lm64" ]]; then
    $l_cluster --text $w2input --c 64 --output_dir clusters/w2lm-64
elif [[ "$1" == "w2lm32" ]]; then
    $l_cluster --text $w2input --c 32 --output_dir clusters/w2lm-32
elif [[ "$1" == "w2lm16" ]]; then
    $l_cluster --text $w2input --c 16 --output_dir clusters/w2lm-16
elif [[ "$1" == "w2lm10" ]]; then
    $l_cluster --text $w2input --c 10 --output_dir clusters/w2lm-10
elif [[ "$1" == "w2lm9" ]]; then
    $l_cluster --text $w2input --c 9 --output_dir clusters/w2lm-9
elif [[ "$1" == "w2lm8" ]]; then
    $l_cluster --text $w2input --c 8 --output_dir clusters/w2lm-8
elif [[ "$1" == "w2lm4" ]]; then
    $l_cluster --text $w2input --c 4 --output_dir clusters/w2lm-4
elif [[ "$1" == "w2flm256" ]]; then
    $l_cluster --text $w2finput --c 256 --output_dir clusters/w2flm-256
elif [[ "$1" == "w2flm128" ]]; then
    $l_cluster --text $w2finput --c 128 --output_dir clusters/w2flm-128
elif [[ "$1" == "w2flm64" ]]; then
    $l_cluster --text $w2finput --c 64 --output_dir clusters/w2flm-64
elif [[ "$1" == "w2flm32" ]]; then
    $l_cluster --text $w2finput --c 32 --output_dir clusters/w2flm-32
elif [[ "$1" == "w2flm16" ]]; then
    $l_cluster --text $w2finput --c 16 --output_dir clusters/w2flm-16
elif [[ "$1" == "w2flm10" ]]; then
    $l_cluster --text $w2finput --c 10 --output_dir clusters/w2flm-10
elif [[ "$1" == "w2flm9" ]]; then
    $l_cluster --text $w2finput --c 9 --output_dir clusters/w2flm-9
elif [[ "$1" == "w2flm8" ]]; then
    $l_cluster --text $w2finput --c 8 --output_dir clusters/w2flm-8
elif [[ "$1" == "w2flm4" ]]; then
    $l_cluster --text $w2finput --c 4 --output_dir clusters/w2flm-4
elif [[ "$1" == "w2falm128" ]]; then
    $l_cluster --text $w2fainput --c 128 --output_dir clusters/w2falm-128

else
    echo "Improper argument"
fi

