# scikit-learn example 

You will need python3 and docker to run this example

## Steps to build and run this example:

1. Build a container for training the model 
```
  make stop clean build-train-env 
```
2. Train the model 
```
  make train 
```
This will generate model pickel file and schema.

3. Build a Fiddler container 
```
  make package
```

4. Start fiddler runtime 
```
   make run
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
