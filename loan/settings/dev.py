import dj_database_url
from loan.settings.base import *

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-p2ag41ufe()67uw5o18^$o_mb!!izhr4w_i2b2z8hm%x43y&i4"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

DATABASES["default"] = dj_database_url.parse(
    f"sqlite:////{BASE_DIR / BASE_DIR.name}.db",
    conn_max_age=600,
)


STATIC_URL = "static/"

STATIC_ROOT = BASE_DIR / "assets"

STATIC_DIR = BASE_DIR / "loan/assets"

STATICFILES_DIRS = [
    STATIC_DIR,
]

MEDIA_URL = "/media/"

MEDIA_ROOT = BASE_DIR / "media"