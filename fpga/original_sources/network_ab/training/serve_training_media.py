"""Loopback-only media viewer with HTTP byte ranges for reliable video seeking."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import re


class MediaHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining=None
        path=self.translate_path(self.path)
        value=self.headers.get('Range')
        if not value or not os.path.isfile(path):return super().send_head()
        size=os.path.getsize(path);m=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
        if not m or not any(m.groups()):self.send_error(400,'Only a single byte range is supported');return None
        a,b=m.groups()
        if a:start=int(a);end=min(int(b),size-1) if b else size-1
        else:start=max(0,size-int(b));end=size-1
        if start>=size or start>end:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return None
        f=open(path,'rb');f.seek(start);self.remaining=end-start+1
        self.send_response(206);self.send_header('Content-Type',self.guess_type(path))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(self.remaining));self.end_headers();return f

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes');super().end_headers()

    def copyfile(self,source,outputfile):
        try:
            if self.remaining is None:return super().copyfile(source,outputfile)
            while self.remaining:
                data=source.read(min(256*1024,self.remaining))
                if not data:break
                outputfile.write(data);self.remaining-=len(data)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);ap.add_argument('--port',type=int,default=8883);a=ap.parse_args()
    handler=functools.partial(MediaHandler,directory=a.directory)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),handler)
    print(f'MEDIA_SERVER_READY http://127.0.0.1:{a.port}/ root={os.path.abspath(a.directory)}',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
