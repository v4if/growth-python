#! /usr/bin/python2.7
#coding=utf-8

# @Author: v4if
# @Date:   2016-07-27 12:14:18
# @Last Modified by:   v4if
# @Last Modified time: 2016-10-23 14:58:13

import zipfile
import os
import threading
import time

banner=r'''
by atkjest
'''

# zip --encrypt file.zip file
zfile_path = "/root/Advance/growth-python/zip-explode/file.zip"
zfile_to_path = "/root/Advance/growth-python/zip-explode/zfile_to_path"
pwd_file_dir = "/root/Advance/growth-python/zip-explode/wordlist"

class Dicts(object):
	"""pwd dicts api"""
	def __init__(self, pwd_file_dir):
		self.pwd_file_dir = pwd_file_dir
		# for store dict absolute path
		self.pwd_dicts = []
	def scan_dicts(self):
		for root, dirs, files in os.walk(pwd_file_dir, topdown=True):
		    for pwd_txt in files:
		        pwd_path = os.path.join(root, pwd_txt)
		        self.pwd_dicts.append(pwd_path)

class ThreadSched(object):
	"""sched for thread"""
	def __init__(self, zfile_path, topath, dicts, thread_num):
		self.zfile = zipfile.ZipFile(zfile_path)
		self.topath = topath
		self.dict_lists = dicts.pwd_dicts
		self.thread_num = thread_num
		self.while_len = len(self.dict_lists)
		self.step = int(self.while_len / self.thread_num) + 1
	def extract_file(self, i_min, i_max, tag):
		for pwd_path in self.dict_lists[i_min:i_max]:
		        print("[#] {thread_tag} - {path}".format(thread_tag=tag, path=pwd_path))
		        pwd_file = open(pwd_path, "r")
		        for line in pwd_file.readlines():
		        	try:
		        		p_dict = line.strip()
		        		self.zfile.extractall(path=self.topath, pwd=p_dict)

		    			print('[*] Password Found: ' + p_dict)
		    			os._exit(0)
		        	except:
		        		pass
		        	finally:
		        		pwd_file.close()
	def run(self):
		i = 0
		cnt = 1
		thread = []
		while i < self.while_len:
			i_max = i + self.step
			if i_max > self.while_len:
				i_max = self.while_len
			tag = 'thread {thread_tag}'.format(thread_tag = cnt)
			t = threading.Thread(target=self.extract_file, args=(i, i_max, tag))
			t.start()
			thread.append(t)
			i += self.step
			cnt += 1
			time.sleep(0.1)

		for t in thread:
			# 主调线程堵塞，直到被调用线程运行结束或超时
			t.join()

def main():
	dicts = Dicts(pwd_file_dir)
	dicts.scan_dicts()
	sched = ThreadSched(zfile_path, zfile_to_path, dicts, 5)
	sched.run()

	print('\n[X] sorry, PassWord NOT Found!!,You can try to creack it with a larger dict')

if __name__ == '__main__':
	main()
