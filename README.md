# fiddler-archive
This repository contains examples of fiddler-archive

## Quick start:

You will need python3 and docker to run this example

1. clone this repo
```
  git clone https://github.com/fiddler-labs/fiddler-archive.git
```
2. install fiddler
```
  pip3 install fiddler
```

3. cd to example dir, eg:
```
cd sklearn
```

5. To train a model and build a Fiddler Archive:
```
   make stop clean build run
```

5. check if the server is running:
```
   make logs
```

6. To call predict on a row at specifed index:  
```
   make execute index=20
```

 7. To explan an infrence:
```
   make explain index=20
```
