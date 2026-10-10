# Sabor Express

Study project on object-oriented Python and simple REST APIs, built around a fictional restaurant app.

> Code written in Portuguese (course material).

## Structure

```
sabor-express/
├── oop/                       # Object-oriented domain model
│   ├── main.py                # Demo: builds a restaurant and prints its menu
│   └── modelos/
│       ├── restaurante.py     # Restaurant: ratings, active status, menu
│       ├── avaliacao.py       # Rating
│       └── cardapio/          # Menu items: abstract base class + Prato, Bebida, Sobremesa
└── api/
    ├── main.py                # FastAPI app serving restaurant menus
    ├── fetch_restaurants.py   # Downloads menus from a public API and splits them per restaurant
    └── restaurantes/          # Output of fetch_restaurants.py
```

**Concepts practiced:** classes and encapsulation, `@property`, `@classmethod`, abstract base classes (`abc`), inheritance and polymorphism (`aplicar_desconto` defines a different discount for each menu item type), FastAPI routing and query parameters.

## How to run

```bash
pip install -r requirements.txt

# OOP demo
cd oop && python main.py

# API (docs at http://127.0.0.1:8000/docs)
cd api && uvicorn main:app --reload
```
