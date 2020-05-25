#!/usr/bin/env python3

from subprocess import Popen, PIPE
import json
import os
from tqdm.auto import tqdm
from pathlib import Path
import argparse

def command(cmd):
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    formatted_stdout = '},{'.join(stdout.decode().split('}{'))
    stdout = '[{}]'.format(formatted_stdout)
    return json.loads(stdout)

def listFiles(repo):
    files = command(['pachctl', 'list', 'file', '{}@master'.format(repo), '--raw'])
    return files

def getFile(repo, file, out):
    cmd = ['pachctl', 'get', 'file', '{repo}@master:{file}'.format(repo=repo,file=file), '-o', out]
    command(cmd)

def getFiles(repo, out):
    out_dir = Path(out, repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = listFiles(repo)
    for file in tqdm(files):
        filepath = file['file']['path']
        outpath = out_dir / filepath.split('/').pop()
        getFile(repo, filepath, str(outpath.resolve()))

    print('Files saved: ', len(files))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)

    args = parser.parse_args()

    getFiles(args.repo, args.out)
