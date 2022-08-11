# Dalle-At-Your-Service(ays): Lightweight Dalle Server using FastAPI

# Development
## Locally
Run:
```
uvicorn main:app --host 0.0.0.0 --reload
```

## Docker
Build the image:
```
task build
```

Run:
```
task run
```

Development (live reloads docker builds):
```
task dev -w
```

Lint:
```
task lint
```



