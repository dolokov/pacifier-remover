
"""
    source https://github.com/deepanprabhu/duckduckgo-images-api/blob/master/api.py
"""

import requests;
import re;
import os
import json;
import pprint;
import time;
import urllib

import argparse
import numpy as np 

def search(keywords, directory, max_results=None):
    url = 'https://duckduckgo.com/';
    params = {
    	'q': keywords
    };

    #   First make a request to above URL, and parse out the 'vqd'
    #   This is a special token, which should be used in the subsequent request
    res = requests.post(url, data=params)
    searchObj = re.search(r'vqd=(\d+)\&', res.text, re.M|re.I);

    headers = {
    'dnt': '1',
    'accept-encoding': 'gzip, deflate, sdch, br',
    'x-requested-with': 'XMLHttpRequest',
    'accept-language': 'en-GB,en-US;q=0.8,en;q=0.6,ms;q=0.4',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'referer': 'https://duckduckgo.com/',
    'authority': 'duckduckgo.com',
    }

    params = (
    ('l', 'wt-wt'),
    ('o', 'json'),
    ('q', keywords),
    ('vqd', searchObj.group(1)),
    ('f', ',,,'),
    ('p', '2')
    )

    requestUrl = url + "i.js";

    while True:
        res = requests.get(requestUrl, headers=headers, params=params);
        data = json.loads(res.text);
        printJson(data["results"],directory);
        if "next" not in data:
            exit(0);
        requestUrl = url + data["next"];
        time.sleep(5);

def download_file(url,file_path):
    testfile = urllib.URLopener()
    testfile.retrieve(url,file_path)


def printJson(objs,directory):
    for obj in objs:
        print "Width {0}, Height {1}".format(obj["width"], obj["height"]);
        print "Thumbnail {0}".format(obj["thumbnail"]);
        print "Url {0}".format(obj["url"]);
        print "Title {0}".format(obj["title"].encode('utf-8'));
        print "Image {0}".format(obj["image"]);
        print "__________";

        file_path = os.path.join(directory,'%i'%np.random.randint(100000)+obj['image'].split('/')[-1])
        try:
            download_file(obj['image'],file_path)
        except:
            ''


#search("dora coloring pages");

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search',required=True)
    parser.add_argument('--directory',required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.directory): os.makedirs(args.directory)
    search(args.search,args.directory)