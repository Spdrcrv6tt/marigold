#!/usr/bin/env bash
# LibriSpeech train-clean-360 (CC BY 4.0, openslr.org/12, no account) for the student's audio-teacher distillation.
cd /d/marigold/data/librispeech-360
curl -sS -L -C - --retry 5 -o train-clean-360.tar.gz https://us.openslr.org/resources/12/train-clean-360.tar.gz >> ../logs/ls360.log 2>&1 || \
curl -sS -L -C - --retry 5 -o train-clean-360.tar.gz https://www.openslr.org/resources/12/train-clean-360.tar.gz >> ../logs/ls360.log 2>&1
ls -la train-clean-360.tar.gz >> ../logs/ls360.log
tar -xzf train-clean-360.tar.gz >> ../logs/ls360.log 2>&1 && rm -f train-clean-360.tar.gz
echo "ls360 finished $(date) $(ls LibriSpeech/train-clean-360 | wc -l) speakers" >> ../logs/ls360.log
