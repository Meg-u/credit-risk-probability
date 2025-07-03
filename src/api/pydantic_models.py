from pydantic import BaseModel
from typing import List, Optional

class CustomerData(BaseModel):
    CustomerId: str
    Amount: float
    Value: float
    ProductCategory: str
    ChannelId: str
    TransactionStartTime: str  # ISO format string!

class CustomersRequest(BaseModel):
    customers: List[CustomerData]
