#!/bin/bash
# Run pipeline with default.yaml config using nohup
# Output will be saved to pipeline_nohup.log

cd /root/unlearning_evaluation-1
nohup python pipeline.py --config-name=default > pipeline_nohup.log 2>&1 &
echo "Pipeline started with PID: $!"
echo "Logs will be written to: pipeline_nohup.log"
echo "Monitor progress with: tail -f pipeline_nohup.log"
