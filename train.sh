# https://github.com/tum-bgd/compression-in-dl/blob/Publication-at-IEEE-JSTARS/01_train/train.sh
if [[ "$(docker images -q qc4rs 2> /dev/null)" == "" ]]; then # https://stackoverflow.com/questions/30543409/how-to-check-if-a-docker-image-with-a-specific-tag-exist-locally
    cd container
    make
    cd ..
fi

yourfilenames=`ls configs`

mkdir -p logs

for eachfile in $yourfilenames
do
    #extension="${eachfile##*.}"
    #runid="$(basename ${eachfile} .${extension})"
    #runid=(${runid//_/ })
    #runid=${runid[0]}
    
    echo "=============================="
    echo "START:$(date)" #| tee -a log/logfile.${runid}
    echo "PWD:$(pwd)" #| tee -a log/logfile.${runid}
    echo "HOST:$(hostname)" #| tee -a log/logfile.${runid}

    docker run -it --rm -v $PWD/../../:/tf -w /tf/qc4rs_dir --gpus=all --name=qc4rs_container_test qc4rs /bin/bash -c "python ./qc4rs/entrypoint.py $1/${eachfile}" # 2>&1 | tee -a log/logfile.${runid}"
    # docker run --rm -it --gpus all --user=$(id -u):$(id -g) -v $PWD/../../:/tf -w /tf qc4rs /bin/bash -c "source ./set_env.sh; python3 entrypoint.py $1/${eachfile} 2>&1 | tee -a log/logfile.${runid}"
    #CONTAINER_ID=$(docker run --rm -it --gpus all --user=$(id -u):$(id -g) -v $PWD/../:/tf -w /tf/rsModels_v2 --env HDF5_USE_FILE_LOCKING=FALSE rsmodels/tf_train /bin/bash -c "source ./set_env.sh; python3 entrypoint.py ${eachfile} 2>&1")
    #echo "Container ID: ${CONTAINER_ID}"
    #docker logs $CONTAINER_ID --follow
    echo "END:$(date)" #| tee -a log/logfile.${runid}
    echo "=============================="
done