#/usr/bin/env python3

import glob
import os

def main():
	li = glob.glob('*.lib')
	with open('deps.txt','w') as f:
		for s in li:
			f.write('depReq\\' + s + '\n')
    
if __name__ == "__main__":
    main()
    os.system('start deps.txt')