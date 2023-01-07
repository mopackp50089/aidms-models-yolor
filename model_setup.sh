#!/bin/bash
function error_handling () {
    if [ $? -ne 0 ]
    then
        echo "Error Occurred, Terminating..."
    fi
}

function usage {
        echo "Usage: bash $(basename $0) [-th]" 2>&1
        echo '   -t|--tag   Build a docker image with your argument tag. The default tag will get the value from config.yaml if there is no input argument'
        echo '   -h|--help   Shows usage'
        exit 1
}


function docker_build()
{
	echo "Build image via DockerFile......"
    echo "Building image ---> aidms.com:8443/library/object_detection__yolor:$build_tag"
	docker build -t aidms.com:8443/library/object_detection__yolor:$build_tag .
}

function compile_with_gpu()
{
	echo "Compile mmdet with gpu......"
	docker run -id --gpus=all --name aidms_model_for_compile aidms.com:8443/library/object_detection__yolor:$build_tag
	
	docker exec aidms_model_for_compile bash -c "cd customized/models/SOLO && pip install -v -e . && echo 'ubuntu' | sudo -S rm /usr/bin/gcc && echo 'ubuntu' | sudo -S apt-get update && echo 'ubuntu' | sudo -S apt-get install gcc-4.8 -y && echo 'ubuntu' | sudo -S ln -s /usr/bin/gcc-4.8 /usr/bin/gcc && pip install uwsgi && echo 'ubuntu' | sudo -S ln -sf /home/ubuntu/.local/bin/tensorboard /usr/local/bin/tensorboard"

}


function commit_new_image()
{
	echo "Commit new docker images and replace the old one......"
	docker commit aidms_model_for_compile aidms.com:8443/library/object_detection__yolor:$build_tag
}

function remove_old_container()
{
	docker stop aidms_model_for_compile
	docker rm aidms_model_for_compile
}

function push2harbor()
{
    echo "Check if 'aidms.com' has been correctly set in /etc/hosts......"
    if [ -z $(awk '/aidms.com/{print $1}' /etc/hosts) ] ; then
        echo "aidms.com not set correctly in /etc/hosts !"
        echo "Please upload the model image to Harbor manually after completing the setup in /etc/hosts"
        exit 1
    else
        echo "aidms.com has been correctly set in /etc/hosts !"
        echo "Start pushing to harbor...."
        docker push aidms.com:8443/library/object_detection__yolor:$build_tag
    fi
    
}

set -e
# before leave this script do error_handling
trap error_handling EXIT

# echo original parameters=[$@]
# ARGS=`getopt -o ab:c:: --long help,along,blong:,clong:: -n "$0" -- "$@"`
# -o|--options accept short option，ex: ab:c::，accept option -a -b -c
# ":" means: -a =  does not accept args input, -b: = require args, -c:: = no require args(optional)
# -l|--long accept long option, use "," to seprate
# -n option followed by the option parsing error prompts the name of the script

ARGS=`getopt -o ht: --long help,tag: -n "$0" -- "$@"`
if [ $? != 0 ]; then
    echo "Terminating..."
    usage
    exit 1
fi

# echo ARGS=[$ARGS]
eval set -- "${ARGS}"
# echo formatted parameters=[$@]


while true
do
    case "$1" in
        -h|--help) 
            usage
            shift
            ;;
        -t|--tag)
            args_tag=$2
            echo "Option tag, get argument $2";
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            usage
            exit 1
            ;;

    esac
done

# process remaining parameters
# echo remaining parameters=[$@]
# echo \$1=[$1]
# echo \$2=[$2]


# echo args_tag = $args_tag
# echo ${#args_tag}


source parse_yaml.sh
echo "Get configuration from config.yaml...."
parse_yaml config.yaml
create_variables config.yaml
echo "Done"
# echo $build_tag
# echo $model_compile


# if length of args_tag is 0, use default value from config.yaml
if [ "${#args_tag}" = 0 ]
then
    echo "No tag args input, use default tag from config.yaml"
    # [ "$build_tag" = "\"value\"" ]
    split_tag=(${build_tag//\"/ }) #split by "\""
    # printf "${split_tag[0]}\n"
    build_tag="${split_tag[0]}"
else
    echo "args input tag get $args_tag, replacing $build_tag with $args_tag"
    build_tag=$args_tag
fi
readonly build_tag



# config.yaml format check
if [ $model_compile != true ] && [ $model_compile != false ]; then
    echo "Error: Get model_compile= "$model_compile ", which must in boolean (true or false)"
    exit 1
fi 

if [ $model_compile = true ]
then
    echo "model_compile==true, start building model..."
    docker_build
    compile_with_gpu
    commit_new_image
    remove_old_container
    push2harbor
else
     echo "model_compile==false, start building model..."
    docker_build
    push2harbor
fi
