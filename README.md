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
   make stop clean train build
```

6. Start the server 
```
   make run
```


7. check if the server is running:
```
   make logs
```

8. To call predict on a row at specifed index:  
```
   make execute index=20
```

 9. To explan an infrence:
```
   make explain index=20
```
