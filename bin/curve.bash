#!/bin/bash

plot(){

    if [ "$1"x = "type"x ]; then
        echo "plot PR curve image with type........"
        python get_sparse_pr.py $1
        python bag_plot_tpr.py
    elif [ "$1"x = "no-type"x ]; then
        echo "plot PR curve image without type........"
        python get_sparse_pr.py $1
        python bag_plot_npr.py
    else
        echo "plot PR curve image for multiple parameters........"
        python bag_plot_mpr.py
    fi
}

cd ../scripts

while getopts ":tnmh" opt; do
    case $opt in
        t)
            plot "type"
            ;;
        n)
            plot "no-type"
            ;;
        m)
            plot "multi"
            ;;
        h)
            echo -e "\n Usage: Determine which mode to use"
            echo -e "\t -t\tProcess results data with entity types."
            echo -e "\t -n\tProcess results data without entity type."
            echo -e "\t -m\tProcess result data with multiple parameters."
            echo -e "\t -h\tGet help infomation."
            ;;
        \?)
            echo -e "\n Invalid option: $OPTARG"
            echo -e "\n Try option -h to get help infomation"
            ;;
    esac
done
