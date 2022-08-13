# Dalle-At-Your-Service(ays): Lightweight Dalle Server using FastAPI

# Configuration
Note any env vars with a `None` default value are needed to run the server.

| Variable         | Default                                    | Type |
| ---------------- | ------------------------------------------ | ---- |
| DALLE_MODEL      | "dalle-mini/dalle-mini/mega-1-fp16:latest" | str  |
| VQGAN_REPO       | "dalle-mini/vqgan_imagenet_f16_16384"      | str  |
| VQGAN_COMMIT_ID  | "e93a26e7707683d349bf5d5c41c5b0ef69b677a9" | str  |
| DEFAULT_NUM_PICS | 3                                          | int  |
| WANDB_API_KEY    | None                                       | str  |

# Development

Mandatory env vars are required from the configuration section in a .env file!

To kickstart the creation of that file, run

```
task env
```

## Locally

Configure vscode debugger for live reload local build (you will need cuda and nvidia drivers):

```
{
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["app.main:app"],
      "jinja": true,
      "justMyCode": true,
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}

```

## Docker

Or, alternatively,

Build the image:

```
task build
```

Run:

Using a gpu:
```
task run
```
Using cpu only
```
task run-cpu
```

Development (live reload docker builds with gpu):
Using gpu:
```
task dev -w
```
Using cpu:
```
task dev-cpu -w
```

Lint:

```
task lint
```
