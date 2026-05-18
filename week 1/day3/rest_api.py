from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    id: int
    name:str
    price: int
    is_offer: Optional[bool]=None

@app.get("/")
def message():
    return {"message": "FastAPI working successfully"}

@app.get("/items/{item_id}")
def fetch_item(item_id:int, item:Item, q:Optional[str]=None):
    return {"message": f"Item with id {item_id} fetched successfully."}

@app.post("/items/")
def create_item(item:Item):
    return {"message": f"Item with id {item.id} created successfully."}

@app.delete("/items/{item_id}")
def delete_item(item:Item, q:Optional[str]=None):
    return {"message": f"Item with id {item.id} deleted successfully."}

@app.put("/items/{item_id}")
def update_item(item:Item):
    return {"message":f"Item with id {item.id} updated successfully."}

@app.patch("/items/{item_id}")
def update_partially_item(item:Item):
    return {"message": f"Item with id {item.id} updated partially successfully."}
