#! /usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys


def calc_code(file_path):
    raw_file = open(file_path, 'r')
    code_line = 0
    for raw_line in raw_file:
        code_line += 1
    raw_file.close()
    return code_line


def walk_dir(dir_root, level, strict, total_lines=0):
    if level == 0:
        print dir_root

    for f in os.listdir(dir_root):
        path = os.path.join(dir_root, f)
        if os.path.isdir(path):
            if strict:
                # for php
                (filepath, filename) = os.path.split(f)
                if filename in ['phpMyAdmin']:
                    continue

            total_lines += walk_dir(path, level + 1, strict)
        elif os.path.isfile(path):
            (shortname, extension) = os.path.splitext(f)
            if extension in ['.cpp', '.c', '.java', '.python', '.py', '.php']:
                if strict:
                    # for java
                    if extension == '.java' and shortname in ['BuildConfig', 'R']:
                        continue

                lines = calc_code(path)
                total_lines += lines
                print '  ' * level + '--' + f + '  ' + str(lines)
        else:
            print 'The file type can not be resolved ！'
    return total_lines


def usage():
    print '''
        [usage] python showcode dirpath strictflag
        [info] dirpath must be absolute path
               strictflag must be in [False | True]
        [out] total code lines
    '''
    sys.exit()


def main():
    if len(sys.argv) < 2:
        usage()

    dirpath = sys.argv[1]
    if not os.path.isdir(dirpath):
        usage()

    if len(sys.argv) >= 3:
        raw_arg = sys.argv[2]
        if raw_arg not in ['False', 'True']:
            usage()

        if raw_arg == 'False':
            strict = False
        elif raw_arg == 'True':
            strict = True
    else:
        strict = False

    total_lines = walk_dir(dirpath, 0, strict)

    print "\nThe total code lines is %d\nCongratulation and Come On！" % int(total_lines)


if __name__ == '__main__':
    main()
