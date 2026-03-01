from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    DATABASE_URL: str
    OPENAI_API_KEY: str | None = None
    ENV: str = "prod"

settings = Settings()