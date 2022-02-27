#!/bin/bash
#

PID_FILE="log/gunicorn.pid"
source /etc/profile
start() {
  if [ ! -f $PID_FILE ];then
    gunicorn -c gunicorn_config.py nlp_server:app
    if [ -f $PID_FILE ];then
      echo "The process is started"
    fi
  else
    PID=`cat log/gunicorn.pid`
    echo "Proces is already running with PID: $PID"
  fi
}

stop() {
  if [ ! -f $PID_FILE ];then
    echo "The Process is not running."
  else
    PID=`cat log/gunicorn.pid`
    kill -QUIT $PID
    sleep 5
    if [ ! -f $PID_FILE ];then
      echo "The process is stopped"
    fi
  fi
}

case $1 in
start)
  start
;; 
stop)   
  stop
;; 
restart)
  stop
  sleep 10
  start
;;
status)
  if [ ! -f $PID_FILE ];then
    echo "The Process is not running."
  else
    PID=`cat log/gunicorn.pid`
    echo "The Process Running with PID: $PID"
  fi
    
;; 
esac    
exit 0

