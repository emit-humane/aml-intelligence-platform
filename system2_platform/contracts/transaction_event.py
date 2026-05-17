from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TransactionType = Literal["NEFT", "RTGS", "IMPS", "Wire", "UPI", "Card"]
PaymentChannel = Literal["Mobile", "Web", "ATM", "Branch"]
TransactionStatus = Literal["Success", "Failed", "Pending"]


class TransactionEvent(BaseModel):
    transaction_id: str
    timestamp: datetime
    sender_account: str
    receiver_account: str
    sender_name: str
    receiver_name: str
    sender_bank: str
    receiver_bank: str
    sender_country: str
    receiver_country: str
    amount: float
    currency: str
    transaction_type: TransactionType
    payment_channel: PaymentChannel
    device_id: str
    ip_address: str
    geo_latitude: float
    geo_longitude: float
    merchant_category: str
    transaction_status: TransactionStatus
    sender_balance_before: float
    sender_balance_after: float
    receiver_balance_before: float
    receiver_balance_after: float
    kyc_level: int = Field(ge=0, le=3)
    is_international: bool
    remarks: str
    amount_leading_digit: int = Field(ge=1, le=9)
