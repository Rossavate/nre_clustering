#!/bin/bash

# variables
no_type=false
helping=false
cpu=false
omega=0.001
pattern=15

while getopts ":cnho:p:" opt; do
    case $opt in
        c)
            cpu=true
            ;;
        n)
            no_type=true
            ;;
        o)
            omega=$OPTARG
            ;;
        p)
            pattern=$OPTARG
            ;;
        h)
            echo -e "\n Usage: Determine which mode to use "
            echo -e "\t -c\tRun code only with CPU."
            echo -e "\t -p\tThe number of relation patterns."
            echo -e "\t -n\tRun train and eval without entity type or not."
            echo -e "\t -o\tSet l2_regularizer weight, defualt(0.001)."
            echo -e "\t -h\tGet help infomation."
            helping=true
            ;;
        \?)
            echo -e "\n Invalid option: $OPTARG"
            echo -e "\n Try option -h to get help infomation"
            ;;
    esac
done

if [ $helping = false ]; then
    cd ../src/bag-level

    if [ $cpu = true ]; then
        echo "only use cpu"
        export CUDA_VISIBLE_DEVICES=
    fi

    if [ $no_type = true ]; then
        echo "start evaluating......."
        python eval.py --use_types=False --model=model-$pattern-notype --pattern_num=$pattern --l2_reg_omega=$omega

    else
        echo "start evaluating......."
        python eval.py --model=model-$pattern --pattern_num=$pattern --l2_reg_omega=$omega
    fi
fi
