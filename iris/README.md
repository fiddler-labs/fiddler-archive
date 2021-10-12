# iris example 

In this example model training is done else where.

## Steps to build and run this example:

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
