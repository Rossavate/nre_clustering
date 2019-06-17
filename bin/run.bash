#!/bin/bash

# variables
init=false
no_type=false
helping=false
cpu=false
epoch=1
omega=0.001
pattern=15

while getopts ":icnho:p:" opt; do
    case $opt in
        i)
            init=true
            ;;
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
            echo -e "\t -i\tPrepare all origin txt data."
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

    if [ $init = true ]; then
        echo "staring initialize........"
        python init_data.py
    fi

    if [ $cpu = true ]; then
        echo "only use cpu"
        export CUDA_VISIBLE_DEVICES=
    fi

    if [ $no_type = true ]; then
        for var in 5 10 15 20 25 30 35 40 45 50 60 70 80 90 100;
        do
            for ((i=1; i<=$epoch; i ++))  
            do
                echo "start training........"
                python3 train.py --use_types=False --model=model-$var-notype --pattern_num=$var --l2_reg_omega=$omega
            done
            echo "start evaluating......."
            python3 eval.py --use_types=False --model=model-$var-notype --pattern_num=$var --l2_reg_omega=$omega
        done

    else
        for var in 5 10 15 20 25 30 35 40 45 50 60 70 80 90 100;
        do
            echo pattern_num=$var
            for ((i=1; i<=$epoch; i ++))  
            do
                echo "start training........"
                python3 train.py --model=model-$var --pattern_num=$var --l2_reg_omega=$omega
            done
            echo "start evaluating......."
            python3 eval.py --model=model-$var --pattern_num=$var --l2_reg_omega=$omega
        done
    fi
fi
