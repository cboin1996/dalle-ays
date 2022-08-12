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

Configure vscode debugger:

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
