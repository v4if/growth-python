#!/usr/bin/python2.7
#coding=utf-8

# @Author: v4if

'''
passwd 32位，比hash之后的md5值多一位，去掉一位后暴力测试
'''

import requests
import re
from bs4 import BeautifulSoup

def get_pwn(md5_hash):
    print "正在解密:", md5_hash
    url = "http://cmd5.la"
    payload = {
        'pwd': md5_hash,
        'submit': 'MD5解密',
        'jiejia': 'jie'}
    r = requests.post(url, data=payload)
    r.encoding = 'utf-8'
    html_doc = r.text
    bs = BeautifulSoup(html_doc, 'lxml')
    req = bs.find("div", {"id": "tip"})

    pattern = re.compile(r'<p>.*:(.*?)</p>')
    pwd_passwd = pattern.findall(str(req))
    for i in pwd_passwd:
        if i == "":
            pass
        else:
            print '[+] 原始密码：', i
            return 0
    return 1


if __name__ == '__main__':
    miss_pwd = 'cca9cc444e64c8116a30la00559c042b4'
    for i in range(0, 33):
        md5_hash = '%s%s' % (miss_pwd[0:i], miss_pwd[i+1:])
        req = get_pwn(md5_hash)
        if req == 0:
            exit(0)

    print '[#] 抱歉,没有找到原始密码'