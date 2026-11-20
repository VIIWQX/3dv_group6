#!/usr/bin/env python3
"""Read a bounded artifact chunk for authenticated repository publication."""
import argparse
import base64
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file',type=Path)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--length',type=int,default=49152)
    args=parser.parse_args()
    if args.offset<0 or not 0<args.length<=49152:raise ValueError('Invalid chunk bounds')
    with args.file.open('rb') as stream:stream.seek(args.offset);data=stream.read(args.length)
    print(json.dumps({'offset':args.offset,'bytes':len(data),'base64':base64.b64encode(data).decode()}))

if __name__=='__main__':main()
