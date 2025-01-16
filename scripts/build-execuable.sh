#!/bin/bash

rm -rf dist && rm -rf build

pyinstaller --name iCAP-Device-Data-Analyzer --onefile main.py
