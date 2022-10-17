# -*- coding : utf-8-*-
# coding:unicode_escape
import bayes
import re

if __name__ == '__main__':
    regEx = re.compile(r'\W')

    # test
    emailTest = open('email/ham/1.txt').read()
    listofdic = re.split(r'\W+', emailTest)
    print(listofdic)

    bayes.spamTest()
