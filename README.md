# AI Sales Co-Pilot

A simple MVP for Indian small businesses to upload sales data and get practical insights such as:

- which weekday performs worst
- which products generate the highest margin
- what should be restocked soon
- a basic 7-day revenue forecast using linear regression

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input format

Required columns:

- `date`
- `product`
- `quantity`
- `revenue`

Optional columns:

- `cost`
- `stock`

The app also understands common aliases such as `qty`, `sales`, `item`, and `inventory`.

## Example insights

- "Sales are weakest on Tuesday"
- "Cookies delivers the highest total margin"
- "Restock Cold Coffee soon"

## Next upgrades

- add LLM-generated plain-English recommendations
- segment by store or city
- add anomaly detection for sudden drops
- connect to WhatsApp or email alerts
