# your_app/middleware.py
from rest_framework_simplejwt.tokens import UntypedToken
from rest_framework_simplejwt.authentication import JWTAuthentication
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser
from channels.middleware import BaseMiddleware
from django.db import close_old_connections
import jwt

@database_sync_to_async
def get_user(validated_token):
    try:
        return JWTAuthentication().get_user(validated_token)
    except Exception:
        return AnonymousUser()

class JWTAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        headers = dict(scope['headers'])
        close_old_connections()
        if b'authorization' in headers:
            token_name, token_key = headers[b'authorization'].decode().split()
            if token_name == 'Bearer':
                try:
                    validated_token = UntypedToken(token_key)
                    scope['user'] = await get_user(validated_token)
                except jwt.InvalidTokenError:
                    scope['user'] = AnonymousUser()
        return await super().__call__(scope, receive, send)
