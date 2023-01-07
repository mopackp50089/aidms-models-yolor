TRA_LOGFILE=training.log
INF_LOGFILE=inference.log

fun_date()
{
    while IFS= read -r line; do
    printf '%s %s\n' "$(date)" "$line";
    done
}

displaytime()
{
  local T=$1
  local D=$((T/60/60/24))
  local H=$((T/60/60%24))
  local M=$((T/60%60))
  local S=$((T%60))
  (( $D > 0 )) && printf '%d days ' $D
  (( $H > 0 )) && printf '%d hours ' $H
  (( $M > 0 )) && printf '%d minutes ' $M
  (( $D > 0 || $H > 0 || $M > 0 )) && printf 'and '
  printf '%d seconds\n' $S
}

check_break()
{
  if [ $? != 0 ]; then
    exit 0
  fi
}

training()
{
  # set -x
  start=$(date +%s)
  python3 /workspace/aidms/lib/exec/remove_pre_model.py $1
  check_break
  python3 /workspace/customized/tools/training.py
  check_break
  python3 /workspace/aidms/lib/exec/inference.py
  end=$(date +%s)
  displaytime $end-$start > /workspace/aidms/results/model_$1/training_time.txt
}

inference()
{
  set -x
  pid=$(ps aux | grep inference_restfulapi.py | awk '{print $2}' | sed -n 1p)
  kill -9 $pid
  python3 /workspace/parameters/template_transformer.py inference
  python3 /workspace/models/inference_restfulapi.py
}

if [ -z $1 ]
then
  echo "Please using '-h' option for list of the options"
  exit 0
else
  while [ -n "$1" ]; do
        case "$1" in
	-t) 
	    echo "training"
	    log_id=`cat /workspace/aidms/results/parameters_cluster.yaml | grep select_result_id | awk {'print $2'}`
      python3 /workspace/aidms/lib/exec/run_tensorboard.py ${log_id} 1
	    log_name="${TRA_LOGFILE}"
	    training $log_id 2>&1 | fun_date | tee /workspace/aidms/results/model_${log_id}/log/$log_name
      cp -r /workspace/customized/results/tensorboard/* /workspace/aidms/results/model_${log_id}/tensorboard
      python3 /workspace/aidms/lib/exec/control_weights.py 0 ${log_id}
      python3 /workspace/aidms/lib/exec/write_progress.py ${log_id}
	    break
	    ;;
	-i) 
	    echo "inference" 
	    inference 2>&1 | fun_date | tee -a $INF_LOGFILE
            break	    
	    ;;
	-h)
	    echo "List of the options:"
	    echo " -t : training model."
	    echo " -i : model inference."
	    echo " -h : list of the options."
	    break
            ;;
	*) 
	    echo "Please using '-h' option for list of the options" ;;
	    esac
	    shift
  done
fi
