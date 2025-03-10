from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class RawInputData(BaseModel):
    TransactionId: int
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    CurrencyCode: str
    CountryCode: str
    ProductId: int
    ChannelId: int
    TransactionStartTime: datetime
    Amount: float
    FraudResult: Optional[int] = 0  # Add FraudResult as optional, default to 0