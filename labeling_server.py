from __future__ import print_function

"""
sudo docker run -v /data:/data -v /home/ubuntu:/root -p 0.0.0.0:80:80 -p 0.0.0.0:43928:43928 roomaro python vrnow/servers/room_labeling/server.py

"""
import sys
sys.path.append('/root')
import time
import urlparse
import os 
from glob import glob 
import shutil

import json
import tornado.websocket
import tornado.web
import tornado.httpserver
from tornado.web import asynchronous
from tornado.ioloop import IOLoop
from tornado import gen

import Settings as SE

directory_base,directory_unlabeled,directory_labeled,directory_pacifier,directory_wo_pacifier,directory_rejected = \
    [SE.directory_base,SE.directory_unlabeled,SE.directory_labeled,SE.directory_pacifier,SE.directory_wo_pacifier,SE.directory_rejected]

for d in [directory_base,directory_unlabeled,directory_labeled,directory_pacifier,directory_wo_pacifier,directory_rejected]:
    if not os.path.isdir(d): os.makedirs(d)

DOMAIN = 'localhost'
PORT = 8888

def get_next_unlabeled_url():
    fns = []
    for e in SE.file_extensions:
        fns += glob(os.path.join(directory_unlabeled,'*%s'%e))
    fn = fns[0]
    url = '/images/%s' % fn.split('/')[-1]
    print('[*] served new url',url)
    return url

class LabelingHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('labeling.html',port=PORT,domain=DOMAIN)
     
class DataPointWebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('[WS] opened')
        data = {'url':get_next_unlabeled_url()}
        #data_json = json.dump(data)
        self.write_message(data)

    def on_message(self,message):
        # parse data object
        print('[on_message]',message)
        data = json.loads(message)
        
        # move unlabeled file to labeled clicked directory
        i = int(data['clicked'][-1])
        directory_to_move = [directory_pacifier,directory_wo_pacifier,directory_rejected][i]
        file_path = os.path.join(directory_unlabeled,data['url'].split('/')[-1])
        shutil.move(file_path,directory_to_move)

        # send new job url
        data = {'url':get_next_unlabeled_url()}
        self.write_message(data)

def main(run_locally=False):
    application = tornado.web.Application([
        (r"/label", LabelingHandler),
        (r"/ws_label", DataPointWebSocketHandler),
        (r'/images/(.*)', tornado.web.StaticFileHandler, {'path': directory_unlabeled}) 
    ])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(PORT)
    print('[*] server started on',DOMAIN,PORT)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()        
