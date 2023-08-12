import os
from fastapi import FastAPI, Depends, status, HTTPException
from fastapi.security.http import HTTPBearer, HTTPAuthorizationCredentials
from alpaca import Alpaca, InferenceRequest
from pydantic import BaseModel


class UserStats(BaseModel):
    num_of_requests: int = 0

class ModelResponse(BaseModel):
    input_token_length: int
    n_tokens_truncated: int
    output: str
    output_token_length: int
    reached_max_content_size: int
    total_predict_time_us: int
    total_token_length: int

class ErrorResponse(BaseModel):
    detail: str


global_model: Alpaca | None = None
keys: dict[str, UserStats] | None = None

def get_model() -> Alpaca:
    assert global_model is not None
    return global_model


def get_keys() -> dict[str, UserStats]:
    assert keys is not None
    return keys


def verify_token(token: str) -> bool:
    return token in get_keys()

def update_stats(authorization: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> None:
    try:
        token = authorization.credentials
        get_keys()[token].num_of_requests += 1
    except KeyError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


tags_metadata = [
    {
        "name": "prompts",
        "description": "HTTP endpoints for making prompts to an Alpaca LLM instance."
    }
]

app = FastAPI(openapi_tags=tags_metadata)

@app.on_event("startup")
async def startup_event():
    global global_model, keys
    global_model = Alpaca(os.environ["ALPACA_CLI_PATH"], os.environ["ALPACA_MODEL_PATH"])

    keys = {}
    with open(os.environ["KEYS_PATH"], 'r', encoding='UTF-8') as file:
        while line := file.readline():
            keys[line.rstrip()] = UserStats()


@app.on_event("shutdown")
def shutdown_event():
    global global_model
    if global_model is not None:
        global_model.stop()
        global_model = None


@app.get("/stats", include_in_schema=False)
def stats() -> dict[str, UserStats]:
    return get_keys()


@app.post(
    "/advertisement-mock",
    summary="Mock advertisement prompt",
    description="Get an instant response compatible to the standard model output.",
    tags=["prompts"]
)
def run(_: InferenceRequest) -> ModelResponse:
    return ModelResponse(
        input_token_length=55,
        n_tokens_truncated=0,
        output="Experience the thrill of watching TV like never before with our 4K OLED TV. Get a crystal clear picture and an incredibly thin design, perfect for any home decor!",
        output_token_length=41,
        reached_max_content_size=0,
        total_predict_time_us=24023165,
        total_token_length=96
    )

@app.post(
    "/advertisement",
    summary="Advertisement prompt",
    description="Get an ad text based on the given keywords.",
    tags=["prompts"],
    dependencies=[Depends(update_stats)], responses={
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse}
    }
)
def run(input: InferenceRequest) -> ModelResponse:
    return ModelResponse(**get_model().run(input.wrap_with_advertisement_prompt()))


@app.post(
    "/custom-prompt",
    summary="Custom prompt",
    description="Get a response to any command.",
    tags=["prompts"],
    dependencies=[Depends(update_stats)], responses={
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse}
    }
)
def run(input: InferenceRequest) -> ModelResponse:
    return ModelResponse(**get_model().run(input.wrap_with_default_prompt()))