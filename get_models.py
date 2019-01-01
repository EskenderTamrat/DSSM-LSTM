import os
import requests

location = os.path.normpath('data')
data = {
  'train': { 'file': 'train.pair.tok.ctf' },
  'val':{ 'file': 'valid.pair.tok.ctf' },
  'query': { 'file': 'vocab_Q.wl' },
  'answer': { 'file': 'vocab_A.wl' }
}

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

if __name__ == "__main__":
    if not os.path.exists(location):
        os.mkdir(location)
         
    for item in data.values():
        path = os.path.normpath(os.path.join(location, item['file']))

        if os.path.exists(path):
            print("Reusing locally cached:", path)
            
        else:
            print("Starting download:", item['file'])
            url = "http://www.cntk.ai/jup/dat/DSSM/%s.csv"%(item['file'])
            print(url)
            download(url, path)
            print("Download completed")
        item['file'] = path