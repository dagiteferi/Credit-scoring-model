# credit_scoring_app/schemas.py
from pydantic import BaseModel

class RawInputData(BaseModel):
    TransactionId: int
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    CurrencyCode: str  # e.g., "USD" or "EUR"
    CountryCode: str   # e.g., "US" or "FR"
    ProductId: int     # e.g., 1 or 2
    ChannelId: int     # e.g., 1 or 2
    TransactionStartTime: str  # e.g., "2023-01-01 12:00:00"
    Amount: float