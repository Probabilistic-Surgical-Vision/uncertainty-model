#!/bin/bash

find . -not \( -path "./.git*" -or -path "./.gradient*" \) | xargs rm -rf