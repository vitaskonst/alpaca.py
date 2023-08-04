import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from alpaca import Alpaca, InferenceRequest
from pydantic import BaseModel

class UserStats(BaseModel):
    num_of_requests: int = 0

global_model: Optional[Alpaca] = None
keys: Optional[Dict[str, UserStats]] = None

app = FastAPI()


def get_model() -> Alpaca:
    assert global_model is not None
    return global_model


def get_keys() -> Dict[str, UserStats]:
    assert keys is not None
    return keys


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


class DetailedHTTPException(HTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = "Server error"

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(status_code=self.STATUS_CODE, detail=self.DETAIL, **kwargs)


class DetailedUnauthorizedException(DetailedHTTPException):
    STATUS_CODE = status.HTTP_401_UNAUTHORIZED
    DETAIL = "Unauthorized"

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(headers={"WWW-Authenticate": "Bearer"}, **kwargs)

class NoAuthorizationHeader(DetailedUnauthorizedException):
    DETAIL = "No Authorization header provided"

class InvalidAuthorizationHeader(DetailedUnauthorizedException):
    DETAIL = "Authorization header is invalid. The structure should be: \"Authorization: Bearer <token>\""

class InvalidAuthenticationScheme(DetailedUnauthorizedException):
    DETAIL = "The only authentication scheme supported is Bearer"

class InvalidToken(DetailedUnauthorizedException):
    DETAIL = "The authentication token is invalid"


def verify_token(token: str) -> bool:
    return token in get_keys()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    secured_urls = {'/advertisement', '/custom-prompt'}
    
    if request.url.path in secured_urls:
        try:
            if 'Authorization' not in request.headers:
                raise NoAuthorizationHeader()
            try:
                auth_type, token = request.headers['Authorization'].split()
            except ValueError as err:
                raise InvalidAuthorizationHeader()
            
            if auth_type != 'Bearer':
                raise InvalidAuthenticationScheme()
            
            if not verify_token(token):
                raise InvalidToken()
                
            get_keys()[token].num_of_requests += 1

        except DetailedHTTPException as exc:
            return JSONResponse(status_code=exc.STATUS_CODE, content={'detail': exc.DETAIL}, headers=exc.headers)

    return await call_next(request)

@app.get("/stats", include_in_schema=False)
def stats() -> Dict[str, UserStats]:
    return get_keys()


@app.post("/advertisement-mock")
def run(input: InferenceRequest) -> Dict:
    return {
        "input_token_length": 55,
        "n_tokens_truncated": 0,
        "output": "Experience the thrill of watching TV like never before with our 4K OLED TV. Get a crystal clear picture and an incredibly thin design, perfect for any home decor!",
        "output_token_length": 41,
        "reached_max_content_size": 0,
        "total_predict_time_us": 24023165,
        "total_token_length": 96
    }


@app.post("/advertisement")
def run(input: InferenceRequest) -> Dict:
    return get_model().run(input.wrap_with_advertisement_prompt())


@app.post("/custom-prompt")
def run(input: InferenceRequest) -> Dict:
    return get_model().run(input.wrap_with_default_prompt())