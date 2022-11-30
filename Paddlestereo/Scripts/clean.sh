#!/bin/bash
echo $"Start to clean the project"
rm -r Result
rm -r ResultImg
rm -r log
rm *.log
find . -iname "*.pyc" -exec rm -rf {} \;
find . -iname "*__pycache__*" -exec rm -rf {} \;
echo $"Finish"
