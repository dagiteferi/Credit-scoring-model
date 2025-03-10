from pydantic import BaseModel

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
    TransactionStartTime: str
    Amount: float